"""Delta trees: a code is the sides of a sequence of hyperplanes, each found at its node by Lloyd's algorithm
running on that depth's decoder, which predicts both sides' means.

Pixels are grey levels' bin centers, (level + 0.5) / 256, centered on the training mean. A node's predicted
mean is that mean plus its delta history, one delta per token, and a remainder is an image minus its node's
predicted mean. A hyperplane (n, b), |n| = 1, splits a node into its rare side {n . x >= b} and its common
side, in remainder coordinates; b >= 0 keeps the node's predicted mean on the common side.

Given a node's history and a hyperplane, a depth's decoder predicts the rare side's mean delta r and
m = p / (1 - p) in [0, 1], p being the rare side's share. The common side's mean delta is -m r, so the two
predicted means average to the node's (the tower property). Lloyd's algorithm starts every node of a depth
from the same hyperplane, the depth's start (the training remainders' top principal direction there, through
each node's predicted mean), and each step moves it to the bisector of the two predicted means: normal
r / |r|, bias |r| (1 - m) / 2, which is >= 0 since m <= 1. Every query depends only on the code, so decoding
replays the encoder's choices exactly.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

D = 784
EPS = 1e-8
KEY_BITS = 62  # tokens a node's key holds as plain bits; deeper ones are hashed in

_C1, _C2, _C3 = (c - (1 << 64) for c in (0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB))


def _shr(x, s):
    return (x >> s) & ((1 << (64 - s)) - 1)


def splitmix64(x):
    """A 64-bit integer hash (int64 CPU tensors, wrapping arithmetic)."""
    z = x + _C1
    z = (z ^ _shr(z, 30)) * _C2
    z = (z ^ _shr(z, 27)) * _C3
    return z ^ _shr(z, 31)


def pixels(images):
    """uint8 [N, 28, 28] -> float [N, 784]: each grey level's bin center."""
    return ((images.float() + 0.5) / 256).flatten(1)


def child_key(key, rare, depth):
    """A child's key (int64 on the CPU), which names its code, from its parent's and its token at depth (rare:
    bool): the code's bits while they fit, then a hash of the parent's key and the token, so codes can grow
    without limit."""
    if depth < KEY_BITS:
        return key | (rare.long() << depth)
    return splitmix64(key * 2 + rare.long())


def bisector(r, m):
    """The hyperplane halfway between the predicted means r and -m r: (normal [..., D], bias [...] >= 0)."""
    length = r.norm(dim=-1).clamp_min(EPS)
    return r / length[..., None], length * (1 - m) / 2


def top_direction(R):
    """The top principal direction of remainders R [N, D] (about each node's predicted mean, so pooled within
    nodes), a unit vector [D]."""
    gram = sum((R[i:i + 8192].T @ R[i:i + 8192]).cpu().double() for i in range(0, len(R), 8192))
    assert isinstance(gram, Tensor)
    return torch.linalg.eigh(gram)[1][:, -1].float().to(R.device)


class ImageHead(nn.Module):
    """A feature vector -> channels of a 28 x 28 image."""

    def __init__(self, width, channels):
        super().__init__()
        self.fc = nn.Linear(width, 64 * 7 * 7)
        self.up = nn.Sequential(nn.GELU(), nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.GELU(),
                                nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(), nn.Conv2d(32, channels, 3, padding=1))

    def forward(self, f):
        lead = f.shape[:-1]
        x = self.up(self.fc(f.reshape(-1, f.shape[-1])).view(-1, 64, 7, 7))
        return x.view(*lead, x.shape[1], D)


class Block(nn.Module):
    """Pre-norm attention and feed-forward. Tokens h attend to themselves, or with mem (an earlier layer's
    normalized history tokens), to mem and themselves. Also returns the normalized h, which is what later
    queries read as this layer's memory."""

    def __init__(self, width, heads):
        super().__init__()
        self.ln_a, self.ln_f = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, h, mem=None):
        x = self.ln_a(h)
        kv = x if mem is None else torch.cat([mem, x], 1)
        h = h + self.attn(x, kv, kv, need_weights=False)[0]
        return h + self.ff(self.ln_f(h)), x


class HistoryReader(nn.Module):
    """A node's memory: its history, one token per delta (the delta's direction, log length, side +-1 and m),
    with self-attention among them. A query reads it and itself at every layer (Block with mem), so a node's
    memory is computed once for all its queries."""

    def __init__(self, depth, width, layers, heads):
        super().__init__()
        self.width = width
        self.delta_in = nn.Linear(D + 3, width)
        self.position = nn.Parameter(torch.randn(max(1, depth), width) * 0.02)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))

    def memory(self, deltas, sides, ms):
        """Each layer's normalized history tokens [B, k, W] for histories deltas [B, k, D], sides and ms [B, k]."""
        B, k = sides.shape
        if k == 0:
            return [deltas.new_zeros(B, 0, self.width) for _ in self.blocks]
        length = deltas.norm(dim=-1, keepdim=True).clamp_min(EPS)
        h = self.delta_in(torch.cat([deltas / length, length.log(), sides[..., None], ms[..., None]], -1))
        h = h + self.position[:k]
        mems = []
        for blk in self.blocks:
            h, x = blk(h)
            mems.append(x)
        return mems


class DeltaDecoder(HistoryReader):
    """One depth's decoder: its query is a hyperplane at the node. It also holds the depth's Lloyd start, the
    same at every node (set by the trainer)."""

    scale: Tensor  # the remainders' rms per pixel at this depth: hyperplane biases and r are in its units
    start: Tensor  # [D] the depth's first hyperplane's normal

    def __init__(self, depth, scale=1.0, width=256, layers=4, heads=4):
        super().__init__(depth, width, layers, heads)
        self.config = dict(depth=depth, scale=scale, width=width, layers=layers, heads=heads)
        self.query_in = nn.Linear(D + 1, width)
        self.out_norm = nn.LayerNorm(width)
        self.r_head = ImageHead(width, 1)
        self.m_head = nn.Linear(width, 1)
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.register_buffer("start", torch.zeros(D))

    def query(self, mems, n, b):
        """For a hyperplane n [B, D], b [B] at each of B nodes with memory mems: r [B, D] and m [B]."""
        q = self.query_in(torch.cat([n, (b / self.scale)[:, None]], -1))[:, None]
        for blk, mem in zip(self.blocks, mems):
            q, _ = blk(q, mem)
        f = self.out_norm(q)[:, 0]
        return self.r_head(f)[:, 0] * self.scale, torch.sigmoid(self.m_head(f)[:, 0])


class LeafDecoder(HistoryReader):
    """p(x | leaf): 256 logits per pixel for an image at a leaf, the sum of two paths. One reads the leaf's
    history with a learned query and turns it into an image of logits; the other runs a few convolutions over
    the leaf's predicted mean image, so each pixel's levels can follow its predicted value and neighbours."""

    def __init__(self, depth, width=256, layers=4, heads=4, channels=64):
        super().__init__(depth, width, layers, heads)
        self.config = dict(depth=depth, width=width, layers=layers, heads=heads, channels=channels)
        self.query_token = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.out_norm = nn.LayerNorm(width)
        self.levels_head = ImageHead(width, 256)
        self.mean_path = nn.Sequential(nn.Conv2d(1, channels, 3, padding=1), nn.GELU(),
                                       nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
                                       nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
                                       nn.Conv2d(channels, 256, 1))

    def forward(self, deltas, sides, ms, mu):
        """Logits [B, 256, D] for leaves with histories deltas [B, k, D], sides and ms [B, k] and predicted
        means mu [B, D] (pixel units)."""
        q = self.query_token.expand(len(mu), 1, -1)
        for blk, mem in zip(self.blocks, self.memory(deltas, sides, ms)):
            q, _ = blk(q, mem)
        logits = self.levels_head(self.out_norm(q))[:, 0]
        return logits + self.mean_path(mu.view(-1, 1, 28, 28)).flatten(2)


def lloyd(decoder: DeltaDecoder, mems: list[Tensor], steps: int):
    """Yields (n, b, r, m) at each step for B nodes with memory mems: the hyperplane queried ([B, D], [B]) and
    the decoder's answers ([B, D], [B]). The first is the depth's start through each node's predicted mean;
    each next one is the bisector of the last answers, detached."""
    B = len(mems[0])
    n, b = decoder.start.expand(B, D), decoder.start.new_zeros(B)
    for step in range(steps):
        r, m = decoder.query(mems, n, b)
        yield n, b, r, m
        if step + 1 < steps:
            n, b = bisector(r.detach(), m.detach())


def split_sse(r, m, rare_n, rare_sum, all_n, all_sum):
    """sum over sides of n_side |predicted mean delta - actual mean remainder|^2 [B], from each side's image
    count and remainder sum, expanded so that an empty side adds 0."""
    common_n, common_sum = all_n - rare_n, all_sum - rare_sum

    def term(pred, count, total):
        return (count * pred.square().sum(-1) - 2 * (pred * total).sum(-1)
                + total.square().sum(-1) / count.clamp_min(1))

    return term(r, rare_n, rare_sum) + term(-m[:, None] * r, common_n, common_sum)


@dataclass
class Frontier:
    """The nodes at one depth that hold images of a split, and those images."""
    R: Tensor       # [N, D] each image's remainder
    node: Tensor    # [N] its node
    bits: Tensor    # [Nn] int64 on the CPU: each node's key (child_key), which names its code
    deltas: Tensor  # [Nn, k, D] half precision on the CPU: each node's delta history
    sides: Tensor   # [Nn, k] +1 rare, -1 common
    ms: Tensor      # [Nn, k] the m of each split

    @property
    def depth(self):
        return self.sides.shape[1]

    def history(self, nodes, device):
        nodes = nodes.cpu()
        return (self.deltas[nodes].to(device).float(), self.sides[nodes].to(device), self.ms[nodes].to(device))


def root_frontier(X, mean):
    """Every image at the root."""
    return Frontier(R=X - mean, node=torch.zeros(len(X), dtype=torch.long, device=X.device),
                    bits=torch.zeros(1, dtype=torch.int64), deltas=torch.zeros(1, 0, D, dtype=torch.half),
                    sides=torch.zeros(1, 0), ms=torch.zeros(1, 0))


def node_order(fr):
    """The images sorted by node, and each node's first index into that order [Nn + 1]."""
    order = torch.argsort(fr.node, stable=True)
    counts = torch.bincount(fr.node, minlength=len(fr.bits))
    return order, F.pad(counts.cumsum(0), (1, 0))


def chunk_sse(decoder, fr, imgs, steps, select=False):
    """The objective's sum over images imgs of a frontier (all Lloyd steps), and with select, what the last
    step's split does to them: their squared error before and after it and the bits of their tokens."""
    R = fr.R[imgs]
    nodes, inverse = torch.unique(fr.node[imgs], return_inverse=True)
    B = len(nodes)
    mems = decoder.memory(*fr.history(nodes, R.device))
    onehot = F.one_hot(inverse, B).to(R.dtype)
    all_n, all_sum = onehot.sum(0), onehot.T @ R
    sse = R.new_zeros(())
    for n, b, r, m in lloyd(decoder, mems, steps):
        rare = (R * n[inverse]).sum(-1) >= b[inverse]
        W = onehot * rare[:, None]
        sse = sse + split_sse(r, m, W.sum(0), W.T @ R, all_n, all_sum).sum()
    if not select:
        return sse, None
    pred = torch.where(rare[:, None], r[inverse], -m[inverse, None] * r[inverse])
    p = torch.div(m, m + 1)[inverse]
    bits = -torch.where(rare, p, 1 - p).clamp_min(EPS).log2()
    stats = dict(before=R.square().sum(), after=(R - pred).square().sum(), bits=bits.sum())
    return sse, stats


@torch.no_grad()
def choose(decoder, fr, steps, batch):
    """The split of each node after the Lloyd steps, as (n [Nn, D], b [Nn], r [Nn, D], m [Nn])."""
    device = fr.R.device
    nodes = torch.arange(len(fr.bits))
    out = []
    for i in range(0, len(nodes), batch):
        mems = decoder.memory(*fr.history(nodes[i:i + batch], device))
        *_, last = lloyd(decoder, mems, steps)
        out.append(last)
    return tuple(torch.cat(parts) for parts in zip(*out))


@torch.no_grad()
def split(decoder, fr, steps, batch):
    """The next depth's frontier, every node split by its hyperplane and empty children dropped, and each
    image's token cost in bits under the decoder's shares (the rare side's p = m / (1 + m)) [N]."""
    n, b, r, m = choose(decoder, fr, steps, batch)
    side = torch.cat([(fr.R[i:i + batch] * n[fr.node[i:i + batch]]).sum(-1) >= b[fr.node[i:i + batch]]
                      for i in range(0, len(fr.R), batch)])
    p_rare = (m / (1 + m))[fr.node]
    token_bits = -torch.where(side, p_rare, 1 - p_rare).clamp_min(EPS).log2()
    delta = torch.where(side[:, None], r[fr.node], -m[fr.node, None] * r[fr.node])
    children, node = torch.unique(fr.node * 2 + side, return_inverse=True)
    parent, rare = children // 2, (children % 2).bool()
    child_delta = torch.where(rare[:, None], r[parent], -m[parent, None] * r[parent])
    p, rare = parent.cpu(), rare.cpu()
    child = Frontier(R=fr.R - delta, node=node,
                     bits=child_key(fr.bits[p], rare, fr.depth),
                     deltas=torch.cat([fr.deltas[p], child_delta.half().cpu()[:, None]], 1),
                     sides=torch.cat([fr.sides[p], (2 * rare.float() - 1)[:, None]], 1),
                     ms=torch.cat([fr.ms[p], m[parent].cpu()[:, None]], 1))
    return child, token_bits


def descend(decoder, fr, steps, batch):
    """The next depth's frontier (split's, without the token costs)."""
    return split(decoder, fr, steps, batch)[0]
