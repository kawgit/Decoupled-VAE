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
        self.fp16 = False  # (or float16 autocast: V100s' tensor cores; training then needs a GradScaler)
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
        """The network's autocast context: bfloat16 when self.bf16, float16 when self.fp16 (geometry, partitions
        and losses stay float32), else nothing."""
        if self.fp16:
            return torch.autocast(self.position.device.type, dtype=torch.float16)
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
    """An encoder whose query is a hyperplane at a node. Each depth k (a node's history length) may have a
    scale (the remainders' rms per pixel there, hyperplane biases and r then being in its units); subclasses say
    where it comes from. A seeded encoder also predicts each side's top principal direction, which seeds that
    side's own Lloyd start."""

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


def split_sse(r, m, rare_n, rare_sum, all_n, all_sum):
    """sum over sides of n_side |predicted mean delta - actual mean remainder|^2 [B], from each side's image
    count and remainder sum, expanded so that an empty side adds 0."""
    common_n, common_sum = all_n - rare_n, all_sum - rare_sum

    def term(pred, count, total):
        return (count * pred.square().sum(-1) - 2 * (pred * total).sum(-1)
                + total.square().sum(-1) / count.clamp_min(1))

    return term(r, rare_n, rare_sum) + term(-m[:, None] * r, common_n, common_sum)
