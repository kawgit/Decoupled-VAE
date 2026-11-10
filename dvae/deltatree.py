"""Delta trees: a code is the sides of a sequence of hyperplanes, each found at its node by Lloyd's algorithm
running on that depth's encoder, which predicts both sides' means.

Pixels are grey levels' bin centers, (level + 0.5) / 256, centered on the training mean. A node's predicted
mean is that mean plus its delta history, one delta per token, and a remainder is an image minus its node's
predicted mean. A hyperplane (n, b), |n| = 1, splits a node into its rare side {n . x >= b} and its common
side, in remainder coordinates; b >= 0 keeps the node's predicted mean on the common side.

Given a node's history and a hyperplane, a depth's encoder predicts the rare side's mean delta r and
m = p / (1 - p) in [0, 1], p being the rare side's share. The common side's mean delta is -m r, so the two
predicted means average to the node's (the tower property). Lloyd's algorithm starts every node of a depth
from the same hyperplane, the depth's start (the training remainders' top principal direction there, through
each node's predicted mean), and each step moves it to the bisector of the two predicted means: normal
r / |r|, bias |r| (1 - m) / 2, which is >= 0 since m <= 1. Every query depends only on the code, so decoding
replays the encoder's choices exactly.
"""

import contextlib
import math
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
    """uint8 images [N, ...] -> float [N, pixels]: each level's bin center ([N, 28, 28] -> [N, 784])."""
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
    """A feature vector -> channels images of shape (C, H, W) (28 x 28 grey by default), each flattened as
    pixels() flattens: [..., channels, C * H * W]. With channels_last, the convolutions run in that memory layout
    (faster on tensor cores; the same results)."""

    def __init__(self, width, channels, channels_last=False, shape=(1, 28, 28)):
        super().__init__()
        C, H, W = shape
        self.shape, self.channels = shape, channels
        self.fc = nn.Linear(width, 64 * (H // 4) * (W // 4))
        self.up = nn.Sequential(nn.GELU(), nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.GELU(),
                                nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
                                nn.Conv2d(32, channels * C, 3, padding=1))
        self.channels_last = channels_last
        if channels_last:
            self.up.to(memory_format=torch.channels_last)

    def forward(self, f):
        lead = f.shape[:-1]
        C, H, W = self.shape
        x = self.fc(f.reshape(-1, f.shape[-1])).view(-1, 64, H // 4, W // 4)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x = self.up(x)
        return x.reshape(*lead, self.channels, C * H * W)


class Block(nn.Module):
    """Pre-norm attention and feed-forward. Tokens h attend to themselves (with causal=True, each only to itself
    and those before it), or with mem (an earlier layer's normalized history tokens), to mem and themselves. Also
    returns the normalized h, which is what later queries read as this layer's memory. dropout (training only)
    drops attention weights and both branches' outputs."""

    def __init__(self, width, heads, dropout=0.0):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.ln_a, self.ln_f = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)  # its weights; attend does the math
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width),
                                *([nn.Dropout(dropout)] if dropout > 0 else []))  # none: stage 1's speed

    def attend(self, x, kv, causal=False):
        """self.attn(x, kv, kv), without nn.MultiheadAttention's per-call checks (which cost more than the math
        on small batches)."""
        B, L, W = x.shape
        w, b = self.attn.in_proj_weight, self.attn.in_proj_bias
        q = F.linear(x, w[:W], b[:W])
        k, v = F.linear(kv, w[W:], b[W:]).chunk(2, -1)

        def heads(t):
            return t.reshape(B, t.shape[1], self.heads, W // self.heads).transpose(1, 2)

        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), dropout_p=drop, is_causal=causal)
        out = self.attn.out_proj(out.transpose(1, 2).reshape(B, L, W))
        return F.dropout(out, drop) if drop > 0 else out

    def forward(self, h, mem=None, causal=False):
        x = self.ln_a(h)
        kv = x if mem is None else torch.cat([mem, x], 1)
        h = h + self.attend(x, kv, causal and mem is None)
        return h + self.ff(self.ln_f(h)), x


class MeanImage(nn.Module):
    """Features [B, W] of images [B, D], read as 28 x 28 pictures by a small convolutional net."""

    def __init__(self, width, channels=32):
        super().__init__()
        c = channels
        self.net = nn.Sequential(nn.Conv2d(1, c, 3, stride=2, padding=1), nn.GELU(),  # 14 x 14
                                 nn.Conv2d(c, 2 * c, 3, stride=2, padding=1), nn.GELU(),  # 7 x 7
                                 nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.GELU(),
                                 nn.Flatten(), nn.Linear(2 * c * 49, width))

    def forward(self, images):
        return self.net(images.reshape(-1, 1, 28, 28))


class HistoryReader(nn.Module):
    """A node's memory: its history, one token per delta (the delta's direction, log length, side +-1 and m),
    with self-attention among them. A query reads it and itself at every layer (Block with mem), so a node's
    memory is computed once for all its queries. With node_mean, the memory also holds a token for the node's
    predicted mean image (the training mean, the buffer mean, plus the deltas), read by a small convolutional
    net, so the network sees what the node looks like rather than only the steps that led there. With plain, the
    tokens read only each delta (its sides and ms are not shown: an ablation)."""

    mean: Tensor  # [D] the training mean, with node_mean

    def __init__(self, depth, width, layers, heads, node_mean=False, plain=False, pixels=D):
        super().__init__()
        self.width, self.plain, self.D = width, plain, pixels
        self.bf16 = False  # (set by the trainer: run the network under bfloat16 autocast, see amp)
        self.delta_in = nn.Linear(pixels + 3, width)
        self.position = nn.Parameter(torch.randn(max(1, depth), width) * 0.02)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.mean_in = MeanImage(width) if node_mean else None
        self.extra = int(node_mean)  # memory tokens besides the history's
        if node_mean:
            assert pixels == D, "node_mean reads 28 x 28 images"
            self.mean_token = nn.Parameter(torch.randn(width) * 0.02)
            self.register_buffer("mean", torch.zeros(D))

    def amp(self):
        """The network's autocast context: bfloat16 when self.bf16 (geometry, partitions and losses stay float32),
        else nothing."""
        if not self.bf16:
            return contextlib.nullcontext()
        return torch.autocast(self.position.device.type, dtype=torch.bfloat16)

    def memory(self, deltas, sides, ms):
        """Each layer's normalized memory tokens [B, extra + k, W] for histories deltas [B, k, D], sides and ms
        [B, k]: the node's mean image's token (with node_mean), then the history's."""
        with self.amp():
            return self._memory(deltas, sides, ms)

    def _memory(self, deltas, sides, ms):
        B, k = sides.shape
        if self.plain:
            sides, ms = torch.zeros_like(sides), torch.zeros_like(ms)
        length = deltas.norm(dim=-1, keepdim=True).clamp_min(EPS)
        h = self.delta_in(torch.cat([deltas / length, length.log(), sides[..., None], ms[..., None]], -1))
        h = h + self.position[:k]
        if self.mean_in is not None:
            h = torch.cat([(self.mean_in(self.mean + deltas.sum(1)) + self.mean_token)[:, None], h], 1)
        if h.shape[1] == 0:
            return [deltas.new_zeros(B, 0, self.width) for _ in self.blocks]
        mems = []
        for blk in self.blocks:
            h, x = blk(h)
            mems.append(x)
        return mems


class SliceNet(HistoryReader):
    """An encoder whose query is a hyperplane at a node. Each depth k (a node's history length) has a Lloyd
    start, the same at every node of the depth, and may have a scale (the remainders' rms per pixel there,
    hyperplane biases and r then being in its units); subclasses say where they come from. A seeded encoder
    also predicts each side's top principal direction, which seeds that side's own Lloyd start."""

    def __init__(self, depth, width, layers, heads, seeded=False, node_mean=False, plain_history=False,
                 shape=(1, 28, 28)):
        super().__init__(depth, width, layers, heads, node_mean, plain_history, shape[0] * shape[1] * shape[2])
        self.query_in = nn.Linear(self.D + 1, width)
        self.out_norm = nn.LayerNorm(width)
        self.r_head = ImageHead(width, 1, shape=shape)
        self.m_head = nn.Linear(width, 1)
        self.dir_head = ImageHead(width, 2, shape=shape) if seeded else None

    def scale_at(self, k) -> Tensor | float:
        raise NotImplementedError

    def start_at(self, k) -> Tensor:
        raise NotImplementedError

    def query(self, mems, n, b):
        """For a hyperplane n [B, D], b [B] at each of B nodes with memory mems: r [B, D], m [B], and from a
        seeded encoder the rare and common sides' unit directions u [B, 2, D] (else None)."""
        scale = self.scale_at(mems[0].shape[1] - self.extra)  # the nodes' depth
        with self.amp():
            q = self.query_in(torch.cat([n, (b / scale)[:, None]], -1))[:, None]
            for blk, mem in zip(self.blocks, mems):
                q, _ = blk(q, mem)
            f = self.out_norm(q)[:, 0]
            r, m = self.r_head(f)[:, 0].float(), self.m_head(f)[:, 0].float()
            u = None
            if self.dir_head is not None:
                u = self.dir_head(f).float()
        if u is not None:
            u = u / u.norm(dim=-1, keepdim=True).clamp_min(EPS)
        return r * scale, torch.sigmoid(m), u


class Encoder(SliceNet):
    """One encoder for every depth below depths, with each depth's Lloyd start (set by the trainer; a seeded
    one only uses the root's, its other nodes starting from their parent's predicted direction). Its biases
    and deltas are in pixel units, its depth's scale being its own to learn; scaled=True keeps a scale per
    depth, as the first one-stage trees did. grow adds depths: the only parts sized by them are the history
    positions, each drawn from its own seed (so a row is the same whenever it is added), and the starts."""

    starts: Tensor  # [depths, D]
    scales: Tensor  # [depths], when scaled

    def __init__(self, depths, width=256, layers=4, heads=4, scaled=False, seeded=False, node_mean=False,
                 plain_history=False, shape=(1, 28, 28)):
        shape = tuple(shape)
        super().__init__(depths, width, layers, heads, seeded, node_mean, plain_history, shape)
        self.config = dict(depths=depths, width=width, layers=layers, heads=heads, scaled=scaled, seeded=seeded,
                           node_mean=node_mean, plain_history=plain_history, shape=shape)
        self.scaled = scaled
        with torch.no_grad():
            self.position.copy_(self.position_rows(0, len(self.position)))
        self.register_buffer("starts", torch.zeros(depths, self.D))
        if scaled:
            self.register_buffer("scales", torch.ones(depths))

    @property
    def depths(self):
        return self.starts.shape[0]

    def position_rows(self, first, last):
        """History positions first..last - 1 [last - first, W], row i drawn from seed i."""
        rows = [torch.randn(self.width, generator=torch.Generator().manual_seed(i)) for i in range(first, last)]
        return (torch.stack(rows) if rows else torch.zeros(0, self.width)) * 0.02

    @torch.no_grad()
    def grow(self, depths, starts):
        """Extend to depths, the new depths starting from starts [depths - self.depths, D]. The position
        parameter keeps its identity (an optimizer's state for it must grow to match) and loses its gradient."""
        p = self.position
        p.data = torch.cat([p.data, self.position_rows(len(p), max(1, depths)).to(p)])
        p.grad = None
        self.starts = torch.cat([self.starts, starts.to(self.starts)])
        if self.scaled:
            self.scales = torch.cat([self.scales, self.scales.new_ones(depths - len(self.scales))])
        self.config["depths"] = depths

    def scale_at(self, k):
        return self.scales[k] if self.scaled else 1.0

    def start_at(self, k):
        return self.starts[k]


class Decoder(HistoryReader):
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


def lloyd(encoder: SliceNet, mems: list[Tensor], start: Tensor, steps: int, tol: float = 0.0,
          grad: set[int] | None = None):
    """Yields (active, n, b, r, m, u) at each step: the nodes still moving (indices into the B nodes with
    memory mems), the hyperplanes queried there and the encoder's answers. Every node starts from start [B, D]
    through its predicted mean; each next hyperplane is the bisector of the last answers, detached. With
    tol > 0 a node stops once that would move its hyperplane by less than tol (1 - cos of the normals plus the
    bias's change relative to |r|, squared); every node stops after steps queries. With grad, only the steps it
    names (from 0) keep a graph for the backward pass (each query's hyperplane is detached, so no gradient
    crosses from one step to the next)."""
    active = torch.arange(len(start), device=start.device)
    n, b = start, start.new_zeros(len(start))
    for step in range(steps):
        with torch.set_grad_enabled(torch.is_grad_enabled() and (grad is None or step in grad)):
            r, m, u = encoder.query(mems, n, b)
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

    def starts(self, encoder, nodes):
        """Lloyd's first normals [len(nodes), D] at nodes."""
        if self.seed is not None:
            return self.seed[nodes.to(self.seed.device)]
        return encoder.start_at(self.depth).expand(len(nodes), -1)


@dataclass
class Split:
    """Every node's split after Lloyd's steps: its last hyperplane (n [Nn, D], b [Nn]), the encoder's answers
    there (r [Nn, D], m [Nn] and, from a seeded encoder, the sides' directions u [Nn, 2, D]), and the steps it
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
                    deltas=torch.zeros(1, 0, X.shape[1], dtype=torch.half, device=store),
                    sides=torch.zeros(1, 0, device=store), ms=torch.zeros(1, 0, device=store))


def node_order(fr):
    """The images sorted by node, and each node's first index into that order [Nn + 1]."""
    order = torch.argsort(fr.node, stable=True)
    counts = torch.bincount(fr.node, minlength=len(fr.bits))
    return order, F.pad(counts.cumsum(0), (1, 0))


def by_nodes(imgs, node, cap):
    """Cut images grouped by node (imgs, their nodes node) into pieces holding at most cap nodes each: the
    encoder's memory use grows with the nodes it queries, not the images."""
    new = torch.ones_like(node, dtype=torch.long)
    new[1:] = (node[1:] != node[:-1]).long()
    piece = (new.cumsum(0) - 1) // cap
    return list(imgs.split(torch.unique_consecutive(piece, return_counts=True)[1].tolist()))


def tilted_draw(steps: int, draws: int, tilt: float | None, rng: float | None = None):
    """Which of steps Lloyd steps to backpropagate, draws of them without replacement, step i drawn with
    probability proportional to 1 + tilt - i / steps (tilt None: all alike), and each drawn step's weight
    1 / (steps * its chance of being drawn), so the weighted sum of the drawn steps' losses is, in expectation,
    the average over all steps. Systematic sampling makes each chance exact: draws times its share, with any
    step whose chance would pass 1 always drawn and the rest shared among the others. rng: the uniform draw (default
    torch's). Returns {step: weight}."""
    if draws >= steps:
        return {i: 1 / steps for i in range(steps)}
    p = [1.0 if tilt is None else 1 + tilt - i / steps for i in range(steps)]
    chance, sure, rest = [0.0] * steps, set(), list(range(steps))
    while True:  # steps whose chance would pass 1 are always drawn
        left = draws - len(sure)
        total = sum(p[i] for i in rest)
        over = [i for i in rest if left * p[i] >= total]
        if not over:
            break
        sure.update(over)
        rest = [i for i in rest if i not in over]
    left = draws - len(sure)
    for i in sure:
        chance[i] = 1.0
    picked = set(sure)
    if left > 0 and rest:
        total = sum(p[i] for i in rest)
        u = float(torch.rand(())) if rng is None else rng
        edge = 0.0
        for i in rest:  # systematic: the points u, u + 1, ..., each in at most one interval of length < 1
            chance[i] = left * p[i] / total
            lo, edge = edge, edge + chance[i]
            if u + math.ceil(lo - u) < edge:
                picked.add(i)
    return {i: 1 / (steps * chance[i]) for i in picked}


def chunk_sse(encoder, fr, imgs, steps, select=False, tol=0.0, dir_weight=1.0, grad_steps=0,
              grad_tilt: float | None = None, clock=None):
    """The objective's sum over images imgs of a frontier, each node's averaged over its Lloyd steps: the
    split's error and, for a seeded encoder, dir_weight times its sides' rank-1 PCA reconstruction error (each
    image's remainder after the split, less its projection on its side's predicted direction). Also returns
    the reconstruction error's own sum (detached; 0 when not seeded); with select, what the last step's split
    does to the images (their squared error before and after it and the bits of their tokens); the chunk's
    nodes [b]; and their Split (detached). imgs=None takes all of fr, whose nodes are known without asking the
    GPU (so without waiting for it). With 0 < grad_steps < steps (and tol = 0), only that many steps are
    backpropagated, drawn by tilted_draw (grad_tilt; None: uniformly): the objective's value is all of them, its
    gradient the drawn steps' weighted sum (the same in expectation, a cheaper backward pass with bounded
    memory). clock (train_encoder.py's DepthClock) times the memory and the steps' forward pass."""
    if imgs is None:
        R, inverse = fr.R, fr.node
        nodes = torch.arange(len(fr.bits), device=R.device)
    else:
        R = fr.R[imgs]
        nodes, inverse = torch.unique(fr.node[imgs], return_inverse=True)
    B, device = len(nodes), R.device
    began = None if clock is None else clock.start()
    mems = encoder.memory(*fr.history(nodes, device))
    if clock is not None:
        clock.stop(fr.depth, began, "memory")
        began = clock.start()
    all_n = R.new_zeros(B).index_add_(0, inverse, R.new_ones(len(R)))
    P = R.shape[1]  # pixels
    all_sum = R.new_zeros(B, P).index_add_(0, inverse, R)
    node_loss, node_pca, node_steps = R.new_zeros(B), R.new_zeros(B), R.new_zeros(B)
    last_n, last_b, last_r, last_m = R.new_zeros(B, P), R.new_zeros(B), R.new_zeros(B, P), R.new_zeros(B)
    last_u = None
    slot = torch.full((B,), -1, dtype=torch.long, device=device)
    weights = tilted_draw(steps, grad_steps, grad_tilt) if 0 < grad_steps < steps and tol == 0 else None
    grad = None if weights is None else set(weights)
    drawn = R.new_zeros(B)  # with grad, the drawn steps' weighted loss per node
    for i, (active, n, b, r, m, u) in enumerate(lloyd(encoder, mems, fr.starts(encoder, nodes), steps, tol, grad)):
        if len(active) == B:  # every node still moving (always, with tol = 0): no masking, which would wait
            Ra, ia = R, inverse
        else:
            slot.fill_(-1)
            slot[active] = torch.arange(len(active), device=device)
            at = slot[inverse]
            mine = at >= 0
            Ra, ia = R[mine], at[mine]
        rare = (Ra * n[ia]).sum(-1) >= b[ia]
        rare_n = R.new_zeros(len(active)).index_add_(0, ia, rare.to(R.dtype))
        rare_sum = R.new_zeros(len(active), P).index_add_(0, ia, Ra * rare[:, None])
        loss = split_sse(r, m, rare_n, rare_sum, all_n[active], all_sum[active])
        if u is not None:
            child = (Ra - torch.where(rare[:, None], r[ia], -m[ia, None] * r[ia])).detach()
            side_u = torch.where(rare[:, None], u[ia, 0], u[ia, 1])
            pca = child.square().sum(-1) - (child * side_u).sum(-1).square()
            pca = R.new_zeros(len(active)).index_add_(0, ia, pca)
            loss = loss + dir_weight * pca
            node_pca = node_pca.index_add(0, active, pca.detach())
            if last_u is None:
                last_u = R.new_zeros(B, 2, P)
            last_u[active] = u.detach()
        node_loss = node_loss.index_add(0, active, loss if grad is None else loss.detach())
        if weights is not None and i in weights:
            drawn = drawn.index_add(0, active, loss * weights[i])
        node_steps[active] += 1
        last_n[active], last_b[active], last_r[active], last_m[active] = n, b, r.detach(), m.detach()
    if clock is not None:
        clock.stop(fr.depth, began, "steps")
    sse = (node_loss / node_steps).sum()
    if weights is not None:  # every step's value, the drawn steps' gradient
        estimate = drawn.sum()
        sse = sse + estimate - estimate.detach()
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
def choose(encoder, fr, steps, batch, tol=0.0):
    """Every node's Split after the Lloyd steps."""
    device = fr.R.device
    nodes = torch.arange(len(fr.bits))
    parts = []
    for i in range(0, len(nodes), batch):
        part = nodes[i:i + batch]
        mems = encoder.memory(*fr.history(part, device))
        start = fr.starts(encoder, part)
        B = len(part)
        P = start.shape[1]
        n, b, r, m = start.new_zeros(B, P), start.new_zeros(B), start.new_zeros(B, P), start.new_zeros(B)
        u, taken = None, start.new_zeros(B)
        for active, sn, sb, sr, sm, su in lloyd(encoder, mems, start, steps, tol):
            n[active], b[active], r[active], m[active] = sn, sb, sr, sm
            if su is not None:
                u = start.new_zeros(B, 2, P) if u is None else u
                u[active] = su
            taken[active] += 1
        parts.append(Split(n, b, r, m, u, taken))
    return cat_splits(parts)


@torch.no_grad()
def split(encoder, fr, steps, batch, tol=0.0):
    """The next depth's frontier, every node split by its hyperplane and empty children dropped, each image's
    token cost in bits under the encoder's shares (the rare side's p = m / (1 + m)) [N], and the Split."""
    s = choose(encoder, fr, steps, batch, tol)
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


def descend(encoder, fr, steps, batch, tol=0.0):
    """The next depth's frontier (split's, without the token costs or the Split)."""
    return split(encoder, fr, steps, batch, tol)[0]


def tree_steps(ckpt, depths):
    """Each depth's Lloyd steps [depths]: the checkpoint's own per-depth caps when it has them, else the one
    --lloyd-steps it was trained with."""
    caps = ckpt.get("caps")
    return list(caps)[:depths] if caps is not None else [ckpt["args"]["lloyd_steps"]] * depths


def load_tree(ckpt, device):
    """A trained encoder (train_encoder.py's checkpoint) as a list of its depths' encoders: the same
    Encoder at every depth, in eval mode."""
    config = dict(ckpt["shared"]["config"], scaled="scales" in ckpt["shared"]["state_dict"],
                  seeded=any(k.startswith("dir_head.") for k in ckpt["shared"]["state_dict"]))
    encoder = Encoder(**config).to(device)
    encoder.load_state_dict(ckpt["shared"]["state_dict"])
    return [encoder.eval()] * ckpt["shared"]["config"]["depths"]
