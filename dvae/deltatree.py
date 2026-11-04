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
        self.heads = heads
        self.ln_a, self.ln_f = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)  # its weights; attend does the math
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def attend(self, x, kv):
        """self.attn(x, kv, kv), without nn.MultiheadAttention's per-call checks (which cost more than the math
        on small batches)."""
        B, L, W = x.shape
        w, b = self.attn.in_proj_weight, self.attn.in_proj_bias
        q = F.linear(x, w[:W], b[:W])
        k, v = F.linear(kv, w[W:], b[W:]).chunk(2, -1)

        def heads(t):
            return t.reshape(B, t.shape[1], self.heads, W // self.heads).transpose(1, 2)

        out = F.scaled_dot_product_attention(heads(q), heads(k), heads(v))
        return self.attn.out_proj(out.transpose(1, 2).reshape(B, L, W))

    def forward(self, h, mem=None):
        x = self.ln_a(h)
        kv = x if mem is None else torch.cat([mem, x], 1)
        h = h + self.attend(x, kv)
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


class SliceDecoder(HistoryReader):
    """A decoder whose query is a hyperplane at a node. Each depth k (a node's history length) has a Lloyd
    start, the same at every node of the depth, and may have a scale (the remainders' rms per pixel there,
    hyperplane biases and r then being in its units); subclasses say where they come from. A seeded decoder
    also predicts each side's top principal direction, which seeds that side's own Lloyd start."""

    def __init__(self, depth, width, layers, heads, seeded=False):
        super().__init__(depth, width, layers, heads)
        self.query_in = nn.Linear(D + 1, width)
        self.out_norm = nn.LayerNorm(width)
        self.r_head = ImageHead(width, 1)
        self.m_head = nn.Linear(width, 1)
        self.dir_head = ImageHead(width, 2) if seeded else None

    def scale_at(self, k) -> Tensor | float:
        raise NotImplementedError

    def start_at(self, k) -> Tensor:
        raise NotImplementedError

    def query(self, mems, n, b):
        """For a hyperplane n [B, D], b [B] at each of B nodes with memory mems: r [B, D], m [B], and from a
        seeded decoder the rare and common sides' unit directions u [B, 2, D] (else None)."""
        scale = self.scale_at(mems[0].shape[1])
        q = self.query_in(torch.cat([n, (b / scale)[:, None]], -1))[:, None]
        for blk, mem in zip(self.blocks, mems):
            q, _ = blk(q, mem)
        f = self.out_norm(q)[:, 0]
        u = None
        if self.dir_head is not None:
            u = self.dir_head(f)
            u = u / u.norm(dim=-1, keepdim=True).clamp_min(EPS)
        return self.r_head(f)[:, 0] * scale, torch.sigmoid(self.m_head(f)[:, 0]), u


class DeltaDecoder(SliceDecoder):
    """One depth's decoder, with that depth's scale and Lloyd start (set by the trainer)."""

    scale: Tensor  # the depth's scale
    start: Tensor  # [D] the depth's first hyperplane's normal

    def __init__(self, depth, scale=1.0, width=256, layers=4, heads=4):
        super().__init__(depth, width, layers, heads)
        self.config = dict(depth=depth, scale=scale, width=width, layers=layers, heads=heads)
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.register_buffer("start", torch.zeros(D))

    def scale_at(self, k):
        return self.scale

    def start_at(self, k):
        return self.start


class SharedDecoder(SliceDecoder):
    """One decoder for every depth below depths, with each depth's Lloyd start (set by the trainer; a seeded
    one only uses the root's, its other nodes starting from their parent's predicted direction). Its biases
    and deltas are in pixel units, its depth's scale being its own to learn; scaled=True keeps a scale per
    depth, as the first one-stage trees did."""

    starts: Tensor  # [depths, D]
    scales: Tensor  # [depths], when scaled

    def __init__(self, depths, width=256, layers=4, heads=4, scaled=False, seeded=False):
        super().__init__(depths, width, layers, heads, seeded)
        self.config = dict(depths=depths, width=width, layers=layers, heads=heads, scaled=scaled, seeded=seeded)
        self.scaled = scaled
        self.register_buffer("starts", torch.zeros(depths, D))
        if scaled:
            self.register_buffer("scales", torch.ones(depths))

    def scale_at(self, k):
        return self.scales[k] if self.scaled else 1.0

    def start_at(self, k):
        return self.starts[k]


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


def lloyd(decoder: SliceDecoder, mems: list[Tensor], start: Tensor, steps: int, tol: float = 0.0):
    """Yields (active, n, b, r, m, u) at each step: the nodes still moving (indices into the B nodes with
    memory mems), the hyperplanes queried there and the decoder's answers. Every node starts from start [B, D]
    through its predicted mean; each next hyperplane is the bisector of the last answers, detached. With
    tol > 0 a node stops once that would move its hyperplane by less than tol (1 - cos of the normals plus the
    bias's change relative to |r|, squared); every node stops after steps queries."""
    active = torch.arange(len(start), device=start.device)
    n, b = start, start.new_zeros(len(start))
    for step in range(steps):
        r, m, u = decoder.query(mems, n, b)
        yield active, n, b, r, m, u
        if step + 1 == steps:
            break
        n2, b2 = bisector(r.detach(), m.detach())
        if tol > 0:
            move = 1 - (n2 * n).sum(-1) + ((b2 - b) / r.detach().norm(dim=-1).clamp_min(EPS)).square()
            moving = move >= tol
            if not moving.any():
                break
            if not moving.all():
                active, n2, b2, mems = active[moving], n2[moving], b2[moving], [mem[moving] for mem in mems]
        n, b = n2, b2


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
    """The nodes at one depth that hold images of a split, and those images. The nodes' fields live on the CPU,
    or for a resident frontier (a batch's, small enough) on the images' device."""
    R: Tensor       # [N, D] each image's remainder
    node: Tensor    # [N] its node
    bits: Tensor    # [Nn] int64: each node's key (child_key), which names its code
    deltas: Tensor  # [Nn, k, D] half precision: each node's delta history
    sides: Tensor   # [Nn, k] +1 rare, -1 common
    ms: Tensor      # [Nn, k] the m of each split
    seed: Tensor | None = None  # [Nn, D]: each node's Lloyd start, from a seeded parent (else its depth's start)

    @property
    def depth(self):
        return self.sides.shape[1]

    def history(self, nodes, device):
        nodes = nodes.to(self.deltas.device)
        return (self.deltas[nodes].to(device).float(), self.sides[nodes].to(device), self.ms[nodes].to(device))

    def starts(self, decoder, nodes):
        """Lloyd's first normals [len(nodes), D] at nodes."""
        if self.seed is not None:
            return self.seed[nodes.to(self.seed.device)]
        return decoder.start_at(self.depth).expand(len(nodes), D)


@dataclass
class Split:
    """Every node's split after Lloyd's steps: its last hyperplane (n [Nn, D], b [Nn]), the decoder's answers
    there (r [Nn, D], m [Nn] and, from a seeded decoder, the sides' directions u [Nn, 2, D]), and the steps it
    took [Nn]."""
    n: Tensor
    b: Tensor
    r: Tensor
    m: Tensor
    u: Tensor | None
    steps: Tensor


def root_frontier(X, mean, resident=False):
    """Every image at the root; resident keeps the nodes' fields on the images' device."""
    store = X.device if resident else torch.device("cpu")
    return Frontier(R=X - mean, node=torch.zeros(len(X), dtype=torch.long, device=X.device),
                    bits=torch.zeros(1, dtype=torch.int64, device=store),
                    deltas=torch.zeros(1, 0, D, dtype=torch.half, device=store),
                    sides=torch.zeros(1, 0, device=store), ms=torch.zeros(1, 0, device=store))


def node_order(fr):
    """The images sorted by node, and each node's first index into that order [Nn + 1]."""
    order = torch.argsort(fr.node, stable=True)
    counts = torch.bincount(fr.node, minlength=len(fr.bits))
    return order, F.pad(counts.cumsum(0), (1, 0))


def by_nodes(imgs, node, cap):
    """Cut images grouped by node (imgs, their nodes node) into pieces holding at most cap nodes each: the
    decoder's memory use grows with the nodes it queries, not the images."""
    new = torch.ones_like(node, dtype=torch.long)
    new[1:] = (node[1:] != node[:-1]).long()
    piece = (new.cumsum(0) - 1) // cap
    return list(imgs.split(torch.unique_consecutive(piece, return_counts=True)[1].tolist()))


def chunk_sse(decoder, fr, imgs, steps, select=False, tol=0.0, dir_weight=1.0):
    """The objective's sum over images imgs of a frontier, each node's averaged over its Lloyd steps: the
    split's error and, for a seeded decoder, dir_weight times its sides' rank-1 PCA reconstruction error (each
    image's remainder after the split, less its projection on its side's predicted direction). Also returns
    the reconstruction error's own sum (detached; 0 when not seeded); with select, what the last step's split
    does to the images (their squared error before and after it and the bits of their tokens); the chunk's
    nodes [b]; and their Split (detached)."""
    R = fr.R[imgs]
    nodes, inverse = torch.unique(fr.node[imgs], return_inverse=True)
    B, device = len(nodes), R.device
    mems = decoder.memory(*fr.history(nodes, device))
    all_n = R.new_zeros(B).index_add_(0, inverse, R.new_ones(len(R)))
    all_sum = R.new_zeros(B, D).index_add_(0, inverse, R)
    node_loss, node_pca, node_steps = R.new_zeros(B), R.new_zeros(B), R.new_zeros(B)
    last_n, last_b, last_r, last_m = R.new_zeros(B, D), R.new_zeros(B), R.new_zeros(B, D), R.new_zeros(B)
    last_u = None
    slot = torch.full((B,), -1, dtype=torch.long, device=device)
    for active, n, b, r, m, u in lloyd(decoder, mems, fr.starts(decoder, nodes), steps, tol):
        slot.fill_(-1)
        slot[active] = torch.arange(len(active), device=device)
        at = slot[inverse]
        mine = at >= 0
        Ra, ia = R[mine], at[mine]
        rare = (Ra * n[ia]).sum(-1) >= b[ia]
        rare_n = R.new_zeros(len(active)).index_add_(0, ia, rare.to(R.dtype))
        rare_sum = R.new_zeros(len(active), D).index_add_(0, ia, Ra * rare[:, None])
        loss = split_sse(r, m, rare_n, rare_sum, all_n[active], all_sum[active])
        if u is not None:
            child = (Ra - torch.where(rare[:, None], r[ia], -m[ia, None] * r[ia])).detach()
            side_u = torch.where(rare[:, None], u[ia, 0], u[ia, 1])
            pca = child.square().sum(-1) - (child * side_u).sum(-1).square()
            pca = R.new_zeros(len(active)).index_add_(0, ia, pca)
            loss = loss + dir_weight * pca
            node_pca = node_pca.index_add(0, active, pca.detach())
            if last_u is None:
                last_u = R.new_zeros(B, 2, D)
            last_u[active] = u.detach()
        node_loss = node_loss.index_add(0, active, loss)
        node_steps[active] += 1
        last_n[active], last_b[active], last_r[active], last_m[active] = n, b, r.detach(), m.detach()
    sse = (node_loss / node_steps).sum()
    pca_sum = (node_pca / node_steps).sum()
    split = Split(last_n, last_b, last_r, last_m, last_u, node_steps)
    if not select:
        return sse, pca_sum, None, nodes, split
    rare = (R * last_n[inverse]).sum(-1) >= last_b[inverse]
    pred = torch.where(rare[:, None], last_r[inverse], -last_m[inverse, None] * last_r[inverse])
    p = torch.div(last_m, last_m + 1)[inverse]
    bits = -torch.where(rare, p, 1 - p).clamp_min(EPS).log2()
    stats = dict(before=R.square().sum(), after=(R - pred).square().sum(), bits=bits.sum())
    return sse, pca_sum, stats, nodes, split


def cat_splits(parts: list[Split]) -> Split:
    us = [p.u for p in parts]
    u = None if any(x is None for x in us) else torch.cat([x for x in us if x is not None])
    return Split(torch.cat([p.n for p in parts]), torch.cat([p.b for p in parts]), torch.cat([p.r for p in parts]),
                 torch.cat([p.m for p in parts]), u, torch.cat([p.steps for p in parts]))


@torch.no_grad()
def choose(decoder, fr, steps, batch, tol=0.0):
    """Every node's Split after the Lloyd steps."""
    device = fr.R.device
    nodes = torch.arange(len(fr.bits))
    parts = []
    for i in range(0, len(nodes), batch):
        part = nodes[i:i + batch]
        mems = decoder.memory(*fr.history(part, device))
        start = fr.starts(decoder, part)
        B = len(part)
        n, b, r, m = start.new_zeros(B, D), start.new_zeros(B), start.new_zeros(B, D), start.new_zeros(B)
        u, taken = None, start.new_zeros(B)
        for active, sn, sb, sr, sm, su in lloyd(decoder, mems, start, steps, tol):
            n[active], b[active], r[active], m[active] = sn, sb, sr, sm
            if su is not None:
                u = start.new_zeros(B, 2, D) if u is None else u
                u[active] = su
            taken[active] += 1
        parts.append(Split(n, b, r, m, u, taken))
    return cat_splits(parts)


@torch.no_grad()
def split(decoder, fr, steps, batch, tol=0.0):
    """The next depth's frontier, every node split by its hyperplane and empty children dropped, each image's
    token cost in bits under the decoder's shares (the rare side's p = m / (1 + m)) [N], and the Split."""
    s = choose(decoder, fr, steps, batch, tol)
    child, token_bits = children(fr, s, batch)
    return child, token_bits, s


@torch.no_grad()
def children(fr, s: Split, batch=8192):
    """split's frontier and token costs for every node's Split s; a seeded Split seeds each child with its
    side's predicted direction."""
    n, b, r, m = s.n, s.b, s.r, s.m
    side = torch.cat([(fr.R[i:i + batch] * n[fr.node[i:i + batch]]).sum(-1) >= b[fr.node[i:i + batch]]
                      for i in range(0, len(fr.R), batch)])
    p_rare = (m / (1 + m))[fr.node]
    token_bits = -torch.where(side, p_rare, 1 - p_rare).clamp_min(EPS).log2()
    delta = torch.where(side[:, None], r[fr.node], -m[fr.node, None] * r[fr.node])
    children, node = torch.unique(fr.node * 2 + side, return_inverse=True)
    parent, rare = children // 2, (children % 2).bool()
    child_delta = torch.where(rare[:, None], r[parent], -m[parent, None] * r[parent])
    seed = None if s.u is None else torch.where(rare[:, None], s.u[parent, 0], s.u[parent, 1])
    store = fr.bits.device
    p, rare = parent.to(store), rare.to(store)
    child = Frontier(R=fr.R - delta, node=node,
                     bits=child_key(fr.bits[p], rare, fr.depth),
                     deltas=torch.cat([fr.deltas[p], child_delta.half().to(store)[:, None]], 1),
                     sides=torch.cat([fr.sides[p], (2 * rare.float() - 1)[:, None]], 1),
                     ms=torch.cat([fr.ms[p], m[parent].to(store)[:, None]], 1), seed=seed)
    return child, token_bits


def descend(decoder, fr, steps, batch, tol=0.0):
    """The next depth's frontier (split's, without the token costs or the Split)."""
    return split(decoder, fr, steps, batch, tol)[0]


def load_tree(ckpt, device):
    """A trained tree's decoders, one per depth (the same SharedDecoder at every depth for a one-model tree),
    from its checkpoint (train.py's or train_walk.py's)."""
    if "shared" in ckpt:
        config = dict(ckpt["shared"]["config"], scaled="scales" in ckpt["shared"]["state_dict"],
                      seeded=any(k.startswith("dir_head.") for k in ckpt["shared"]["state_dict"]))
        decoder = SharedDecoder(**config).to(device)
        decoder.load_state_dict(ckpt["shared"]["state_dict"])
        return [decoder.eval()] * ckpt["shared"]["config"]["depths"]
    decoders = []
    for cfg, state in zip(ckpt["configs"], ckpt["decoders"]):
        decoder = DeltaDecoder(**cfg).to(device)
        decoder.load_state_dict(state)
        decoders.append(decoder.eval())
    return decoders
