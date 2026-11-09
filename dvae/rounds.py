"""Rounds through a generator (train_rounds.py): an encoder, a prior and a decoder over codes, each round's encoder
trained on images the last round's generator makes, the generator being that round's encoder, prior and decoder.

A code is a path of tokens in 0..V-1 down a tree whose every node above depth K has V centroids, images the encoder
gives; an image takes the token of its nearest centroid at every depth, and its node at depth K (a leaf) is its
code. The prior and the decoder read a node as its history of choices (PathNet): a causal transformer whose
position 0 is the root and position j the choice made at depth j - 1, read as the V centroids there, each marked
chosen or not, so a node at depth d is d choices, one per token:
- the k-means encoder (CodeEncoder) reads it the same way and gives each node's V centroids, as offsets from the
  centroid chosen last (or, direct, as the data's mean plus its output);
- the Lloyd encoder (LloydEncoder, V = 2) is dvae/deltatree.py's Encoder: its Lloyd steps at a node give a
  hyperplane and the two sides' predicted means, the node's centroids; an image takes its side of the hyperplane;
- the prior (CodePrior) also reads each node's own V centroids, unmarked, and gives the logits of the one chosen;
- the decoder (Decoder) is a pair: its mean (DecoderMean) gives a leaf's mean image as an offset from the centroid
  chosen last, and its covariance (DecoderCov), optional, also reads that mean and gives a Gaussian around it.

The generator makes images (sample_codes): codes drawn from the prior down the encoder's tree, each with an image
from the decoder at its leaf, its mean or a draw from its Gaussian. Sampling runs depth by depth, each model keeping
its state from the depths before (PathNet.step's attention keys and values, the Lloyd encoder's histories), so a
step costs one new position, not the path; and codes sharing a prefix share its nodes' work (the encoder and prior
depend only on the path).
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dvae.deltatree import D, Block, Encoder, ImageHead, children, choose, lloyd, root_frontier


def block_step(blk: Block, h: Tensor, cache: tuple[Tensor, Tensor] | None):
    """blk on one new position h [B, 1, W] of a causal sequence, its earlier positions' attention keys and values
    in cache (each [B, heads, L, W / heads]; None before the first): the output and the cache with this one added.
    The same math as blk(h, causal=True) over the whole sequence, in eval mode."""
    B, _, W = h.shape
    x = blk.ln_a(h)
    w, b = blk.attn.in_proj_weight, blk.attn.in_proj_bias
    q = F.linear(x, w[:W], b[:W])
    k, v = F.linear(x, w[W:], b[W:]).chunk(2, -1)

    def heads(t):
        return t.reshape(B, 1, blk.heads, W // blk.heads).transpose(1, 2)

    q, k, v = heads(q), heads(k), heads(v)
    if cache is not None:
        k, v = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)
    out = F.scaled_dot_product_attention(q, k, v)
    h = h + blk.attn.out_proj(out.transpose(1, 2).reshape(B, 1, W))
    return h + blk.ff(blk.ln_f(h)), (k, v)


def chosen(tok: Tensor, prev: Tensor) -> Tensor:
    """The centroid chosen [..., D] in each choice: tok [...] the token, prev [..., V, D] the choice's centroids (the
    start token, V, reads as token 0: callers mask the root)."""
    V = prev.shape[-2]
    return prev.gather(-2, tok.clamp(max=V - 1)[..., None, None].expand(*tok.shape, 1, prev.shape[-1]))[..., 0, :]


class PathNet(nn.Module):
    """A causal transformer over a node's history of choices, position j being the node at depth j (0: the root,
    K: a leaf). Position 0 reads a start embedding; position j >= 1 reads the choice made at depth j - 1, its V
    centroids each with a mark, 1 if chosen and 0 if not; with candidates=True, every position also reads its node's
    own V centroids, unmarked."""

    def __init__(self, K, V, width=256, layers=4, heads=4, dropout=0.0, candidates=False):
        super().__init__()
        self.V = V
        self.start = nn.Parameter(torch.randn(width) * 0.02)
        self.choice_in = nn.Linear(V * (D + 1), width)
        self.cands_in = nn.Linear(V * D, width) if candidates else None
        self.position = nn.Parameter(torch.randn(K + 1, width) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(width, heads, dropout) for _ in range(layers))
        self.out_norm = nn.LayerNorm(width)

    def embed(self, tok: Tensor, prev: Tensor, cands: Tensor | None, first: int) -> Tensor:
        marks = F.one_hot(tok.clamp(max=self.V - 1), self.V).to(prev.dtype)
        choice = self.choice_in(torch.cat([prev, marks[..., None]], -1).flatten(-2))
        h = torch.where((tok == self.V)[..., None], self.start.to(choice.dtype), choice)
        h = h + self.position[first:first + tok.shape[1]]
        if self.cands_in is not None:
            assert cands is not None
            h = h + self.cands_in(cands.flatten(-2))
        return h

    def features(self, tok: Tensor, prev: Tensor, cands: Tensor | None = None) -> Tensor:
        """Features [B, L, W] of the first L nodes of B paths: tok [B, L] the token chosen in the choice that led to
        each (V at the root), prev [B, L, V, D] that choice's centroids (anything at the root), and cands
        [B, L, V, D] the nodes' own centroids (with candidates)."""
        h = self.drop(self.embed(tok, prev, cands, 0))
        for blk in self.blocks:
            h, _ = blk(h, causal=True)
        return self.out_norm(h)

    def step(self, j: int, tok: Tensor, prev: Tensor, cands: Tensor | None, cache: list | None):
        """Features [B, W] of the nodes at depth j of B paths whose earlier nodes went through step (their keys and
        values in cache, None at the root), given tok [B], prev [B, V, D] and cands [B, V, D] at depth j; and the
        cache with depth j added. Eval mode only."""
        h = self.embed(tok[:, None], prev[:, None], None if cands is None else cands[:, None], j)
        out = []
        for i, blk in enumerate(self.blocks):
            assert isinstance(blk, Block)
            h, kv = block_step(blk, h, None if cache is None else cache[i])
            out.append(kv)
        return self.out_norm(h)[:, 0], out


class CodeEncoder(PathNet):
    mean: Tensor  # [D] the data's mean: the base of direct centroids, and of the root's offsets

    def __init__(self, K, V, width=256, layers=4, heads=4, dropout=0.0, channels_last=False, direct=False,
                 centered=False):
        super().__init__(K, V, width, layers, heads, dropout)
        self.config = dict(K=K, V=V, width=width, layers=layers, heads=heads, dropout=dropout,
                           channels_last=channels_last, direct=direct, centered=centered)
        self.head = ImageHead(width, V, channels_last)
        self.direct, self.centered = direct, centered
        self.register_buffer("mean", torch.zeros(D))

    def base(self, tok: Tensor, prev: Tensor) -> Tensor:
        """What the head's output is added to [..., 1, D] for tok [...] and prev [..., V, D]: the data's mean
        (direct), or the centroid chosen last (the mean at the root)."""
        if self.direct:
            return self.mean.expand(*tok.shape, 1, D)
        return torch.where((tok == self.V)[..., None], self.mean, chosen(tok, prev))[..., None, :]

    def offsets(self, f: Tensor) -> Tensor:
        """The head's V outputs [..., V, D] for features f [..., W]; centered, less their mean, so a node's centroids
        average to its base (no bias: for V = 2 they are base +- r, a tower's m held at 1)."""
        out = self.head(f)
        return out - out.mean(-2, keepdim=True) if self.centered else out

    def centroids(self, tok: Tensor, prev: Tensor, last=False) -> Tensor:
        """The V centroids [B, L, V, D] of the L nodes on B paths (tok, prev as features takes them), or with last
        those of the last node only [B, 1, V, D]."""
        f = self.features(tok, prev)
        if last:
            f, tok, prev = f[:, -1:], tok[:, -1:], prev[:, -1:]
        return self.base(tok, prev) + self.offsets(f)

    def node_step(self, j, tok, prev, cache):
        f, cache = self.step(j, tok, prev, None, cache)
        return self.base(tok, prev) + self.offsets(f), cache


class TableEncoder(nn.Module):
    """Round 0's encoder: centroids table [K, V, D], those of depth j at every node of depth j."""
    table: Tensor

    def __init__(self, table: Tensor):
        super().__init__()
        self.register_buffer("table", table)

    def centroids(self, tok: Tensor, prev: Tensor, last=False) -> Tensor:
        L = tok.shape[1]
        rows = self.table[L - 1:L] if last else self.table[:L]
        return rows.expand(len(tok), -1, -1, -1)

    def node_step(self, j, tok, prev, cache):
        return self.table[j].expand(len(tok), -1, -1), cache


@dataclass
class LloydNodes:
    """LloydEncoder's sampling state at the nodes of one depth: each node's predicted mean mu [n, D], its history as
    dvae/deltatree.py's Frontier holds it (deltas [n, k, D] in half precision, sides and ms [n, k]), and its split's
    answers, the rare side's delta r [n, D] and m [n]."""
    mu: Tensor
    deltas: Tensor
    sides: Tensor
    ms: Tensor
    r: Tensor
    m: Tensor

    def gather(self, idx: Tensor) -> "LloydNodes":
        return LloydNodes(self.mu[idx], self.deltas[idx], self.sides[idx], self.ms[idx], self.r[idx], self.m[idx])


class LloydEncoder(nn.Module):
    """dvae/deltatree.py's Encoder (net) as a rounds encoder, V = 2. At a node, steps Lloyd steps from its depth's
    start give a hyperplane (n, b) and the predicted means of its two sides, the node's centroids: token 0 the
    common side's, mu - m r, and token 1 the rare side's, mu + r, mu being the node's predicted mean. An image takes
    the token of its side of the hyperplane (route), as train_encoder.py's trees do. Each depth's start is the top
    principal direction of the remainders there, tracked while training (train_rounds.py) in grams [depths, D, D]
    and tracked [depths, D]; mean [D] is the data's mean."""
    mean: Tensor
    grams: Tensor
    tracked: Tensor

    def __init__(self, depths, steps, width=256, layers=4, heads=4, node_batch=4096):
        super().__init__()
        self.net = Encoder(depths, width, layers, heads)
        self.config = dict(depths=depths, steps=steps, width=width, layers=layers, heads=heads)
        self.steps, self.V, self.node_batch = steps, 2, node_batch
        self.register_buffer("mean", torch.zeros(D))
        self.register_buffer("grams", torch.zeros(depths, D, D))
        self.register_buffer("tracked", torch.zeros(depths, D))

    def answers(self, j: int, deltas: Tensor, sides: Tensor, ms: Tensor) -> tuple[Tensor, Tensor]:
        """The last Lloyd step's answers at nodes of depth j with histories deltas, sides and ms: r [n, D], m [n]."""
        rs, ms_out = [], []
        for i in range(0, len(sides), self.node_batch):
            mems = self.net.memory(deltas[i:i + self.node_batch].float(), sides[i:i + self.node_batch],
                                   ms[i:i + self.node_batch])
            start = self.net.start_at(j).expand(len(sides[i:i + self.node_batch]), D)
            r = m = None
            for _, _, _, r, m, _ in lloyd(self.net, mems, start, self.steps):
                pass
            assert r is not None and m is not None
            rs.append(r.float())
            ms_out.append(m.float())
        return torch.cat(rs), torch.cat(ms_out)

    def node_step(self, j: int, tok: Tensor, prev: Tensor, cache: LloydNodes | None):
        """The centroids [n, 2, D] of the n nodes at depth j that tok [n] led to from their parents (whose state
        cache holds, gathered to them; None at the root), and the nodes' state. prev is not needed: a node's
        history is in its parent's state."""
        with torch.autocast(self.mean.device.type, enabled=False):
            if cache is None:
                mu = self.mean[None]
                deltas, sides, ms = mu.new_zeros(1, 0, D).half(), mu.new_zeros(1, 0), mu.new_zeros(1, 0)
            else:
                rare = tok == 1
                delta = torch.where(rare[:, None], cache.r, -cache.m[:, None] * cache.r)
                mu = cache.mu + delta
                deltas = torch.cat([cache.deltas, delta.half()[:, None]], 1)
                sides = torch.cat([cache.sides, (2 * rare.float() - 1)[:, None]], 1)
                ms = torch.cat([cache.ms, cache.m[:, None]], 1)
            r, m = self.answers(j, deltas, sides, ms)
            return torch.stack([mu - m[:, None] * r, mu + r], 1), LloydNodes(mu, deltas, sides, ms, r, m)

    @torch.no_grad()
    def route(self, X: Tensor, batch=2048) -> "Tree":
        """Send images X [N, D] down the tree, each to its side of every node's hyperplane: the Tree they reach
        (its centroids each node's two predicted side means, its errors each image's from its side's)."""
        dev, K = X.device, self.net.depths
        fr = root_frontier(X, self.mean, resident=True)
        mu = self.mean[None]
        tree = Tree(parent=[torch.zeros(0, dtype=torch.long, device=dev)],
                    token=[torch.full((1,), 2, dtype=torch.long, device=dev)],
                    node=[torch.zeros(len(X), dtype=torch.long, device=dev)], C=[], mean=self.mean, V=2)
        for _ in range(K):
            s = choose(self.net, fr, self.steps, batch)
            tree.C.append(torch.stack([mu - s.m[:, None] * s.r, mu + s.r], 1).half())
            side = torch.cat([(fr.R[i:i + 8192] * s.n[fr.node[i:i + 8192]]).sum(-1) >= s.b[fr.node[i:i + 8192]]
                              for i in range(0, len(fr.R), 8192)])
            keys = torch.unique(fr.node * 2 + side.long())  # children's order
            fr, _ = children(fr, s)
            parent, rare = keys // 2, (keys % 2).bool()
            tree.parent.append(parent)
            tree.token.append(keys % 2)
            tree.node.append(fr.node)
            tree.err.append(fr.R.square().sum(-1))
            mu = mu[parent] + torch.where(rare[:, None], s.r[parent], -s.m[parent, None] * s.r[parent])
        return tree


class CodePrior(PathNet):
    def __init__(self, K, V, width=256, layers=4, heads=4, dropout=0.0, channels_last=False):  # (no convolutions)
        super().__init__(K, V, width, layers, heads, dropout, candidates=True)
        self.config = dict(K=K, V=V, width=width, layers=layers, heads=heads, dropout=dropout)
        self.head = nn.Linear(width, V)

    def logits(self, tok: Tensor, prev: Tensor, cands: Tensor) -> Tensor:
        """Logits [B, L, V] of the token chosen at each of the L nodes on B paths, given their centroids cands
        [B, L, V, D]."""
        return self.head(self.features(tok, prev, cands))


class DecoderMean(PathNet):
    """The decoder's mean: a leaf's mean image, as an offset from the centroid chosen last, fit by squared error."""

    def __init__(self, K, V, width=256, layers=4, heads=4, dropout=0.0, channels_last=False):
        super().__init__(K, V, width, layers, heads, dropout)
        self.config = dict(K=K, V=V, width=width, layers=layers, heads=heads, dropout=dropout,
                           channels_last=channels_last)
        self.head = ImageHead(width, 1, channels_last)

    def mean_image(self, tok: Tensor, prev: Tensor) -> Tensor:
        """Each path's leaf mean [B, D] (given all K + 1 positions)."""
        out = self.head(self.features(tok, prev)[:, -1]).float()
        return chosen(tok[:, -1], prev[:, -1]) + out[:, 0]


class DecoderCov(PathNet):
    """The decoder's covariance: a Gaussian around a leaf's mean image mu (decoder_mean's, held fixed), covariance F F^T + diag(var): F [D, rank]
    the leaf's main directions of variation and var each pixel's own variance (at least var_floor). It reads the
    leaf's history of choices, as PathNet does, and mu, added to its features at the leaf; fit by maximum likelihood
    (nll), a model apart from decoder_mean so that each has its own training and early stopping."""

    def __init__(self, K, V, width=256, layers=4, heads=4, dropout=0.0, channels_last=False, rank=16,
                 var_floor=1e-4):
        super().__init__(K, V, width, layers, heads, dropout)
        assert rank >= 1, "rank must be at least 1"
        self.config = dict(K=K, V=V, width=width, layers=layers, heads=heads, dropout=dropout,
                           channels_last=channels_last, rank=rank, var_floor=var_floor)
        self.rank, self.var_floor = rank, var_floor
        self.mean_in = nn.Linear(D, width)
        self.head = ImageHead(width, rank + 1, channels_last)

    def outputs(self, tok: Tensor, prev: Tensor, mu: Tensor) -> tuple[Tensor, Tensor]:
        """Each path's leaf's covariance factor [B, D, rank] and per-pixel variance [B, D], given all K + 1
        positions and the leaf's mean mu [B, D]."""
        out = self.head(self.features(tok, prev)[:, -1] + self.mean_in(mu)).float()
        var = self.var_floor + F.softplus(out[:, 0] - 4)  # (about the data's spread at the start)
        return 0.1 * out[:, 1:].transpose(1, 2), var

    def nll(self, tok: Tensor, prev: Tensor, mu: Tensor, X: Tensor) -> Tensor:
        """Each image's negative log-likelihood [B] (nats) under its leaf's Gaussian, mean mu [B, D]."""
        factor, var = self.outputs(tok, prev, mu)
        return -torch.distributions.LowRankMultivariateNormal(mu.float(), factor, var).log_prob(X)

    def sample(self, tok: Tensor, prev: Tensor, mu: Tensor, rows: Tensor) -> Tensor:
        """Images [n, D], one drawn from the Gaussian of each path's leaf rows [n] (indices into the B paths, whose
        means are mu [B, D]), clamped to the pixels' range [0, 1]."""
        factor, var = self.outputs(tok, prev, mu)
        z = torch.randn(len(rows), self.rank, 1, device=mu.device)
        x = mu[rows].float() + (factor[rows] @ z)[..., 0] + var[rows].sqrt() * torch.randn_like(var[rows])
        return x.clamp(0, 1)


@dataclass
class Decoder:
    """The decoder, a pair: its mean (decoder_mean) and, optionally, its covariance around that mean (decoder_cov)."""
    mean: DecoderMean
    cov: DecoderCov | None = None

    def eval(self):
        self.mean.eval()
        if self.cov is not None:
            self.cov.eval()
        return self


@dataclass
class Draw:
    """Codes drawn from the prior (sample_codes): their tokens tok [n, K + 1] (V, the start token, first), an image
    for each from the decoder at its leaf [n, D] (its mean, or a draw from its covariance's Gaussian around
    it), and the drawing encoder's
    centroids along them: node[j] [n] each
    code's node at depth j < K, whose centroids are C[j][node[j]] (C[j] [n_j, V, D], half precision)."""
    tok: Tensor
    leaf: Tensor
    node: list[Tensor]
    C: list[Tensor]

    def __len__(self):
        return len(self.tok)

    def prev(self, idx: Tensor) -> Tensor:
        """The choices along codes idx [B]: each position's choice's centroids [B, K + 1, V, D] (zeros at the
        root), as PathNet reads them."""
        root = self.C[0].new_zeros(len(idx), 1, *self.C[0].shape[1:]).float()
        return torch.cat([root] + [C[n[idx]].float()[:, None] for C, n in zip(self.C, self.node)], 1)


def cat_draws(draws: list[Draw]) -> Draw:
    nodes, offsets = [], [0] * len(draws[0].C)
    for d in draws:
        nodes.append([n + o for n, o in zip(d.node, offsets)])
        offsets = [o + len(C) for o, C in zip(offsets, d.C)]
    return Draw(torch.cat([d.tok for d in draws]), torch.cat([d.leaf for d in draws]),
                [torch.cat(ns) for ns in zip(*nodes)], [torch.cat(Cs) for Cs in zip(*(d.C for d in draws))])


@torch.no_grad()
def sample_codes(encoder, prior: CodePrior, decoder: Decoder, mean: Tensor, n: int, K: int, V: int,
                 batch=32768, bf16=True) -> Draw:
    """n codes drawn from the prior, down the encoder's tree (a Draw), each leaf's image the decoder's mean or,
    with its covariance, a draw from its Gaussian around that mean. All the models in eval mode. Codes are drawn
    batch at a time, and within a batch the encoder and prior run once per distinct node the codes reach (they
    depend only on the path there), the decoder once per distinct leaf; memory grows with batch (a peak of about
    12 GB for 32k codes of 24 tokens at the default sizes, in bfloat16, on an L40S). With bf16, the models run in
    bfloat16 autocast (on CUDA, or with bf16="always" anywhere)."""
    with torch.autocast(mean.device.type, torch.bfloat16, enabled=bf16 == "always" or (bf16 and mean.is_cuda)):
        return cat_draws([sample_shared(encoder, prior, decoder, mean, min(batch, n - i), K, V)[0]
                          for i in range(0, n, batch)])


def _gather(cache, idx):
    """A model's sampling state (PathNet.step's attention cache, or LloydEncoder's LloydNodes) for nodes idx."""
    if cache is None:
        return None
    if isinstance(cache, LloydNodes):
        return cache.gather(idx)
    return [(k[idx], v[idx]) for k, v in cache]


def kmeans_sse(encoder: CodeEncoder, X: Tensor, attraction: float = 0.0) -> Tensor:
    """Images X [B, D] sent down the encoder's own tree, each to its nearest centroid at every depth: their squared
    errors from the centroids they chose, summed over the batch and the depths (the k-means objective, every depth
    at once), with gradients; plus attraction times every image's squared distance to each of its node's V
    centroids, a weak pull of every centroid toward its node's images, so one that wins none still moves toward
    them (at the cost of drawing each centroid toward its node's mean by about 2 attraction / (1 + 2 attraction)
    of the way, for an even split of two). Depth by depth over the distinct nodes the batch reaches, each reading its history as
    the encoder's own choices (gradients stopped there), through PathNet.step's attention cache; no dropout."""
    K, V = int(encoder.config["K"]), encoder.V
    node = torch.zeros(len(X), dtype=torch.long, device=X.device)
    tok = torch.full((1,), V, dtype=torch.long, device=X.device)
    prev = X.new_zeros(1, V, D)
    cache, total = None, X.new_zeros(())
    for j in range(K):
        c, cache = encoder.node_step(j, tok, prev, cache)
        dist = (X[:, None] - c[node].float()).square().sum(-1)
        best = dist.min(1)
        total = total + best.values.sum() + (attraction * dist.sum() if attraction else 0.0)
        keys, node = torch.unique(node * V + best.indices, return_inverse=True)
        parent, tok = keys // V, keys % V
        prev = c[parent].detach().float()
        if j + 1 < K:
            cache = _gather(cache, parent)
    return total


def sample_shared(encoder, prior, decoder: Decoder, mean, B, K, V, leaf_batch=8192) -> tuple[Draw, Tensor]:
    """B codes drawn as sample_codes draws them, depth by depth over the distinct nodes they reach: the Draw, and
    their log-probabilities under the prior [B]."""
    dev = mean.device
    node = torch.zeros(B, dtype=torch.long, device=dev)  # each code's node at the current depth
    tok = torch.full((1,), V, dtype=torch.long, device=dev)  # per distinct node: the token that led there
    prev = mean.new_zeros(1, V, D)  # and the centroids of the choice that led there
    enc_cache = prior_cache = None
    tokens, parents, nodes, tables = [tok], [node], [], []
    logp = torch.zeros(B, device=dev)
    for j in range(K):
        c, enc_cache = encoder.node_step(j, tok, prev, enc_cache)
        f, prior_cache = prior.step(j, tok, prev, c, prior_cache)
        probs = F.softmax(prior.head(f).float(), -1)[node]
        w = torch.multinomial(probs, 1)[:, 0]
        logp += probs.gather(1, w[:, None])[:, 0].log()
        nodes.append(node)
        tables.append(c.half())  # (the children read these rounded values, as training will)
        keys, node = torch.unique(node * V + w, return_inverse=True)
        parent, tok = keys // V, keys % V
        prev = tables[-1][parent].float()
        if j + 1 < K:
            enc_cache, prior_cache = _gather(enc_cache, parent), _gather(prior_cache, parent)
        tokens.append(tok)
        parents.append(parent)
    leaf = torch.empty(B, D, device=dev)  # each code's image, drawn at its leaf
    for i in range(0, len(tok), leaf_batch):  # each distinct leaf's path, walked up from it
        a = torch.arange(i, min(i + leaf_batch, len(tok)), device=dev)
        mine = (node >= i) & (node < i + len(a))
        path_tok, path_prev = [], []
        for j in range(K, 0, -1):
            path_tok.append(tokens[j][a])
            a = parents[j][a]
            path_prev.append(tables[j - 1][a].float())
        path_tok.append(tokens[0][a])
        path_prev.append(mean.new_zeros(len(a), V, D))
        t, p = torch.stack(path_tok[::-1], 1), torch.stack(path_prev[::-1], 1)
        mu, rows = decoder.mean.mean_image(t, p), node[mine] - i
        leaf[mine] = (mu[rows] if decoder.cov is None else decoder.cov.sample(t, p, mu, rows)).float()
    code = torch.stack([tokens[j][n] for j, n in enumerate(nodes + [node])], 1)
    return Draw(code, leaf, nodes, tables), logp


@dataclass
class Tree:
    """The nodes a set of images reaches under an encoder, by depth (0: the root, K: the leaves). At depth d:
    parent[d] [n_d] each node's parent (none at the root), token[d] [n_d] the token that led there (V at the root),
    node[d] [N] each image's node; and above the leaves C[d] [n_d, V, D] each node's centroids (half precision) and
    err[d] [N] each image's squared error from the centroid it chose there."""
    parent: list[Tensor]
    token: list[Tensor]
    node: list[Tensor]
    C: list[Tensor]
    mean: Tensor
    V: int
    err: list[Tensor] = field(default_factory=list)

    def mses(self, images=slice(None)) -> list[float]:
        """The mse per pixel after each token [K] of images (indices or a slice into the tree's)."""
        return [(e[images].mean() / D).item() for e in self.err]

    @property
    def K(self):
        return len(self.token) - 1


def paths(tree: Tree, d: int, nodes: Tensor):
    """The paths to nodes [n] at depth d, as PathNet reads them: the token chosen in the choice that led to each
    node on them [n, d + 1] (V at the root), that choice's centroids [n, d + 1, V, D] (zeros at the root), and the
    nodes [n, d + 1]."""
    toks, prevs, ancs = [], [], []
    a = nodes
    for j in range(d, -1, -1):
        toks.append(tree.token[j][a])
        ancs.append(a)
        if j > 0:
            a = tree.parent[j][a]
            prevs.append(tree.C[j - 1][a].float())
        else:
            prevs.append(tree.mean.new_zeros(len(nodes), tree.V, D))
    return torch.stack(toks[::-1], 1), torch.stack(prevs[::-1], 1), torch.stack(ancs[::-1], 1)


@torch.no_grad()
def route(encoder, X: Tensor, mean: Tensor, K: int, V: int, batch=2048) -> tuple[Tree, list[float]]:
    """Send images X [N, D] down the encoder's tree to depth K: the Tree they reach, and their mse per pixel after
    each token [K] (each image against the centroid it chose; the tree keeps each image's, in err). A LloydEncoder
    routes by its own hyperplanes (LloydEncoder.route)."""
    if isinstance(encoder, LloydEncoder):
        tree = encoder.route(X, batch)
        return tree, tree.mses()
    dev = X.device
    tree = Tree(parent=[torch.zeros(0, dtype=torch.long, device=dev)],
                token=[torch.full((1,), V, dtype=torch.long, device=dev)],
                node=[torch.zeros(len(X), dtype=torch.long, device=dev)], C=[], mean=mean, V=V)
    for d in range(K):
        n = len(tree.token[d])
        C = torch.empty(n, V, D, dtype=torch.half, device=X.device)
        for i in range(0, n, batch):
            idx = torch.arange(i, min(i + batch, n), device=X.device)
            tok, prev, _ = paths(tree, d, idx)
            C[idx] = encoder.centroids(tok, prev, last=True)[:, 0].half()
        tree.C.append(C)
        side, err = [], []
        for i in range(0, len(X), 8192):
            dist = (X[i:i + 8192, None] - C[tree.node[d][i:i + 8192]].float()).square().sum(-1)
            best = dist.min(1)
            side.append(best.indices)
            err.append(best.values)
        tree.err.append(torch.cat(err))
        keys, inverse = torch.unique(tree.node[d] * V + torch.cat(side), return_inverse=True)
        tree.parent.append(keys // V)
        tree.token.append(keys % V)
        tree.node.append(inverse)
    return tree, tree.mses()
