"""A delta tree built from frozen components: one network per (depth, Lloyd step), train_encoder_components.py's.

Once the networks above a step are frozen, everything the step sees is fixed: each image's node, each node's
history and the hyperplane the step is asked about. So a step's network trains on a fixed set of nodes, each
with exact statistics over all of its training images (their count and sum, and each side's under the
hyperplane), and the val mse after any step is measured exactly from the same statistics. Nothing is routed
while a network trains.

A Group is one split's images (train, val or test) at the tree's current depth: their remainders and nodes, and
the history tables the nodes' histories are read from. Table j holds, for every node at depth j + 1, its delta
(r on the rare side, -m r on the common one), its side and its parent's m; a node's ancestors' rows (anc) give
its history. Tables stay on the GPU while they fit (gpu_budget bytes), else in host memory.
"""

import math
import threading
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dvae.deltatree import EPS, Block, SliceNet, bisector, child_key, split_sse

POSITIONS = 512  # history positions every StepNet has (so any two share a state dict's shapes)


class StepNet(SliceNet):
    """One (depth, Lloyd step)'s network: SliceNet with no Lloyd start of its own (the trainer passes each
    query's hyperplane) and pixel-unit biases and deltas. Every StepNet has POSITIONS history positions, so a
    network warm-starts from any other of the same width, layers and heads."""

    def __init__(self, width=128, layers=4, heads=4, plain_history=True, shape=(1, 28, 28), dropout=0.0):
        shape = tuple(shape)
        super().__init__(POSITIONS, width, layers, heads, False, False, plain_history, shape)
        self.config = dict(width=width, layers=layers, heads=heads, plain_history=plain_history, shape=shape,
                           dropout=dropout)
        if dropout > 0:  # (attention weights, both branches' outputs; no parameters, so states stay alike)
            for blk in self.blocks:
                assert isinstance(blk, Block) and isinstance(blk.ff, nn.Sequential)
                blk.dropout = dropout
                blk.ff.append(nn.Dropout(dropout))

    def scale_at(self, k):
        return 1.0


@dataclass
class Stats:
    """Each node's image count, sum of remainders and sum of squared norms, and under a hyperplane its rare
    side's count and sum."""
    count: Tensor   # [Nn]
    total: Tensor   # [Nn, D]
    sq: Tensor      # [Nn]
    rare_n: Tensor | None = None
    rare_sum: Tensor | None = None


@dataclass
class Group:
    """One split's images at the current depth (see the module's docstring)."""
    name: str
    R: Tensor                    # [N, D] remainders (float32, GPU)
    node: Tensor                 # [N] each image's node (GPU)
    keys: Tensor                 # [Nn] int64 node keys (child_key), CPU
    anc: Tensor                  # [Nn, depth] each node's ancestor's row in table j (GPU)
    deltas: list[Tensor] = field(default_factory=list)  # table j: [Nn_{j+1}, D] half, or int8 with scales
    scales: list[Tensor | None] = field(default_factory=list)  # int8 tables' per-row scales [Nn_{j+1}] (else None)
    sides: list[Tensor] = field(default_factory=list)   # [Nn_{j+1}] +1 rare, -1 common
    ms: list[Tensor] = field(default_factory=list)      # [Nn_{j+1}]
    stats: Stats | None = None
    node_cpu: Tensor | None = None  # node on the CPU (the train group's, for building batches)
    window: int = 0  # > 0: histories hold the last window deltas and one token for the sum of those before
    sums: Tensor | None = None  # [Nn, D] each node's sum of deltas (its predicted mean less the training mean)

    @property
    def depth(self):
        return len(self.deltas)

    @property
    def nodes(self):
        return len(self.keys)

    def history(self, nodes: Tensor):
        """The histories (deltas [B, depth, D] float32, sides and ms [B, depth]) of nodes [B] (GPU)."""
        device, B = self.R.device, len(nodes)
        if self.depth == 0:
            z = self.R.new_zeros(B, 0)
            return self.R.new_zeros(B, 0, self.R.shape[1]), z, z
        rows = self.anc[nodes]
        first = max(0, self.depth - self.window) if self.window > 0 else 0
        out = []
        for j in range(first, self.depth):
            table, scale, at = self.deltas[j], self.scales[j], rows[:, j]
            row = table[at.to(table.device)].to(device, non_blocking=True)
            out.append(row if scale is None else row.float() * scale[at, None])
        deltas = torch.stack(out, 1).float()
        sides = torch.stack([self.sides[j][rows[:, j]] for j in range(first, self.depth)], 1)
        ms = torch.stack([self.ms[j][rows[:, j]] for j in range(first, self.depth)], 1)
        if first > 0:  # the deltas before the window, summed into one token (no side or m)
            assert self.sums is not None
            early = self.sums[nodes] - deltas.sum(1)
            deltas = torch.cat([early[:, None], deltas], 1)
            sides, ms = F.pad(sides, (1, 0)), F.pad(ms, (1, 0))
        return deltas, sides, ms


def root_group(name, X, mean, window=0):
    device = X.device
    g = Group(name, X - mean, torch.zeros(len(X), dtype=torch.long, device=device),
              torch.zeros(1, dtype=torch.int64), torch.zeros(1, 0, dtype=torch.long, device=device),
              window=window, sums=X.new_zeros(1, X.shape[1]) if window > 0 else None)
    refresh_stats(g)
    return g


def group_state(g: Group) -> dict:
    """What load_group needs to rebuild g exactly (on the CPU): its remainders, nodes and the history tables its
    window still reads (the older ones are already empty)."""
    cpu = lambda x: None if x is None else x.cpu()  # noqa: E731
    return dict(name=g.name, R=g.R.cpu(), node=g.node.int().cpu(), keys=g.keys, anc=g.anc.int().cpu(),
                deltas=[x.cpu() for x in g.deltas], scales=[cpu(x) for x in g.scales],
                sides=[x.cpu() for x in g.sides], ms=[x.cpu() for x in g.ms], window=g.window, sums=cpu(g.sums))


def load_group(state: dict, device, gpu_budget: float) -> Group:
    """group_state's group, back on device (its tables there while they fit in gpu_budget bytes, as descend
    keeps them)."""
    deltas = []
    for x in state["deltas"]:
        size = x.numel() * x.element_size()
        on_gpu = device.type != "cuda" or torch.cuda.memory_allocated(device) + size < gpu_budget
        deltas.append(x.to(device) if on_gpu else x.pin_memory())
    g = Group(state["name"], state["R"].to(device), state["node"].long().to(device), state["keys"],
              state["anc"].long().to(device), deltas, [None if x is None else x.to(device) for x in state["scales"]],
              [x.to(device) for x in state["sides"]], [x.to(device) for x in state["ms"]], window=state["window"],
              sums=None if state["sums"] is None else state["sums"].to(device))
    refresh_stats(g)
    return g


def refresh_stats(g: Group, chunk=8192):
    Nn, D = g.nodes, g.R.shape[1]
    count = g.R.new_zeros(Nn).index_add_(0, g.node, g.R.new_ones(len(g.R)))
    total, sq = g.R.new_zeros(Nn, D), g.R.new_zeros(Nn)
    for i in range(0, len(g.R), chunk):
        R, node = g.R[i:i + chunk], g.node[i:i + chunk]
        total.index_add_(0, node, R)
        sq.index_add_(0, node, R.square().sum(-1))
    g.stats = Stats(count, total, sq)
    g.node_cpu = g.node.cpu()


def sides_of(g: Group, n: Tensor, b: Tensor, chunk=8192) -> Tensor:
    """Each image's side [N] (True: rare) under its node's hyperplane (n [Nn, D], b [Nn])."""
    return torch.cat([(g.R[i:i + chunk] * n[g.node[i:i + chunk]]).sum(-1) >= b[g.node[i:i + chunk]]
                      for i in range(0, len(g.R), chunk)])


def side_stats(g: Group, n: Tensor, b: Tensor, chunk=8192) -> Stats:
    """g's stats with each node's rare side's count and sum under hyperplanes n, b."""
    assert g.stats is not None
    Nn, D = g.nodes, g.R.shape[1]
    rare_n, rare_sum = g.R.new_zeros(Nn), g.R.new_zeros(Nn, D)
    for i in range(0, len(g.R), chunk):
        R, node = g.R[i:i + chunk], g.node[i:i + chunk]
        rare = ((R * n[node]).sum(-1) >= b[node]).to(R.dtype)
        rare_n.index_add_(0, node, rare)
        rare_sum.index_add_(0, node, R * rare[:, None])
    s = g.stats
    return Stats(s.count, s.total, s.sq, rare_n, rare_sum)


def split_mse(s: Stats, r: Tensor, m: Tensor) -> float:
    """The mse per pixel after every node's split with predicted deltas r on the rare side and -m r on the
    common one, exactly, from the stats: sum |x - r|^2 over the rare side plus |x + m r|^2 over the common."""
    assert s.rare_n is not None and s.rare_sum is not None
    common_sum = s.total - s.rare_sum
    rr = r.square().sum(-1)
    sse = (s.sq - 2 * (r * s.rare_sum).sum(-1) + 2 * m * (r * common_sum).sum(-1)
           + rr * (s.rare_n + (s.count - s.rare_n) * m.square()))
    return sse.sum().item() / (s.count.sum().item() * r.shape[1])


def mse_now(g: Group) -> float:
    """The mse per pixel at g's depth (its remainders' energy)."""
    assert g.stats is not None
    return g.stats.sq.sum().item() / (len(g.R) * g.R.shape[1])


@torch.no_grad()
def answers(net: StepNet, g: Group, n: Tensor, b: Tensor, batch=1024):
    """The network's r [Nn, D] and m [Nn] at every node of g, queried at hyperplanes n, b."""
    net.eval()
    Nn, D = g.nodes, g.R.shape[1]
    r, m = g.R.new_empty(Nn, D), g.R.new_empty(Nn)
    for i in range(0, Nn, batch):
        nodes = torch.arange(i, min(i + batch, Nn), device=g.R.device)
        mems = net.memory(*g.history(nodes))
        r[i:i + len(nodes)], m[i:i + len(nodes)], _ = net.query(mems, n[nodes], b[nodes])
    return r, m


def node_loss(net: StepNet, g: Group, s: Stats, nodes: Tensor, n: Tensor, b: Tensor) -> Tensor:
    """split_sse at nodes [B] (the excess of the split's error over the sides' own scatter, which has the same
    gradient) [B]."""
    assert s.rare_n is not None and s.rare_sum is not None
    mems = net.memory(*g.history(nodes))
    r, m, _ = net.query(mems, n[nodes], b[nodes])
    return split_sse(r, m, s.rare_n[nodes], s.rare_sum[nodes], s.count[nodes], s.total[nodes])


class ImageStream:
    """An endless stream of the n training images' indices: one shuffle of all of them, then another, and so on,
    taken from the front. Every network's epochs read the same stream, so an epoch is a number of images however
    many the dataset holds."""

    def __init__(self, n: int, gen: torch.Generator):
        self.n, self.gen = n, gen
        self.perm, self.pos = torch.randperm(n, generator=gen), 0

    def take(self, k: int) -> Tensor:
        parts = []
        while k > 0:
            if self.pos == self.n:
                self.perm, self.pos = torch.randperm(self.n, generator=self.gen), 0
            part = self.perm[self.pos:self.pos + k]
            self.pos += len(part)
            k -= len(part)
            parts.append(part)
        return torch.cat(parts)

    def state(self) -> dict:
        return dict(perm=self.perm, pos=self.pos)

    def load(self, state: dict):
        self.perm, self.pos = state["perm"], state["pos"]


def epoch_batches(g: Group, images: Tensor, batch: int, max_nodes: int):
    """An epoch's batches over its images (indices into g's): each batch's B images as its distinct nodes, each
    weighted by the share of its images the batch holds, so that sum(weight * split_sse) / B is an unbiased
    estimate of the objective per image; cut into pieces of at most max_nodes nodes. Built on the CPU: a list of
    batches, each a list of (nodes, weights) pieces, on g's device."""
    assert g.node_cpu is not None and g.stats is not None
    count = g.stats.count.cpu()
    out = []
    for idx in images.split(batch):
        nodes, c = torch.unique(g.node_cpu[idx], return_counts=True)
        w = c.float() / count[nodes]
        pieces = [(nodes[i:i + max_nodes].to(g.R.device, non_blocking=True),
                   w[i:i + max_nodes].to(g.R.device, non_blocking=True)) for i in range(0, len(nodes), max_nodes)]
        out.append((pieces, len(idx)))
    return out


@torch.no_grad()
def descend(g: Group, n: Tensor, b: Tensor, r: Tensor, m: Tensor, gpu_budget: float, table_bits: int = 16,
            side: Tensor | None = None):
    """Split every node of g by its hyperplane (n, b) with predicted deltas r and -m r: g moves to the next
    depth. Returns what the store keeps of the split: each image's side, its new node, each new node's parent and
    side. side [N] (True: rare), if given, is each image's side instead of the hyperplane's (sampled codes)."""
    side = sides_of(g, n, b) if side is None else side
    children, node = torch.unique(g.node * 2 + side.long(), return_inverse=True)
    parent, rare = children // 2, (children % 2).bool()
    delta = torch.where(rare[:, None], r[parent], -m[parent, None] * r[parent])
    g.R = g.R - delta[node]
    device = g.R.device
    if table_bits == 8:  # (each row's largest entry at 127; the history reads it back as float)
        scale = delta.abs().amax(-1).clamp_min(1e-12) / 127
        table, size = (delta / scale[:, None]).round().to(torch.int8), 1
        g.scales.append(scale)
    else:
        table, size = delta.half(), 2
        g.scales.append(None)
    on_gpu = device.type != "cuda" or torch.cuda.memory_allocated(device) + delta.numel() * size < gpu_budget
    g.deltas.append(table if on_gpu else table.cpu().pin_memory())
    g.sides.append(2 * rare.float() - 1)
    g.ms.append(m[parent].clone())
    g.anc = torch.cat([g.anc[parent], torch.arange(len(children), device=device)[:, None]], 1)
    if g.window > 0:
        assert g.sums is not None
        g.sums = g.sums[parent] + delta
        old = g.depth - g.window - 1  # (no history reads that table again: free it)
        if old >= 0:
            g.deltas[old] = g.deltas[old].new_zeros(0, delta.shape[1])
    g.keys = child_key(g.keys[parent.cpu()], rare.cpu(), g.depth - 1)
    g.node = node
    refresh_stats(g)
    return dict(side=side.cpu(), node=node.int().cpu(), parent=parent.int().cpu(), rare=rare.cpu())


def next_hyperplane(r: Tensor, m: Tensor):
    return bisector(r, m)


@torch.no_grad()
def top_direction(R: Tensor, chunk=8192) -> Tensor:
    """The top principal direction [D] of remainders R [N, D] (pooled within nodes), signed so that its largest
    entry is positive (eigh's sign is arbitrary and differs between GPUs: a step-0 network asked about the
    mirrored hyperplane has its sides swapped)."""
    gram = sum(R[i:i + chunk].T @ R[i:i + chunk] for i in range(0, len(R), chunk))
    assert isinstance(gram, Tensor)
    gram = gram.double() if gram.device.type == "cuda" else gram.cpu().double()
    v = torch.linalg.eigh(gram)[1][:, -1]
    return (v * v[v.abs().argmax()].sign()).float().to(R.device)


@torch.no_grad()
def orient(net: StepNet, g: Group, start: Tensor, target: float, batch=1024) -> Tensor:
    """start or -start, whichever makes net's split of g (asked about it with bias 0) come closest to the mse
    target: the sign a reused step-0 network was trained with (runs before top_direction's fixed sign took
    theirs from eigh, whose sign depends on the GPU)."""
    best = (math.inf, start)
    for v in (start, -start):
        n, b = v.expand(g.nodes, -1), g.R.new_zeros(g.nodes)
        miss = abs(split_mse(side_stats(g, n, b), *answers(net, g, n, b, batch)) - target)
        if miss < best[0]:
            best = (miss, v)
        if miss <= 1e-3 * target:  # (the right sign is off by rounding only; the wrong one by a depth's gain)
            break
    return best[1]


class Utilization:
    """The GPU's utilization, sampled every second in a thread (NVML); mean(since) averages the samples taken
    since a time.time() mark (None: no NVML, or not CUDA)."""

    def __init__(self, device):
        self.samples: list[tuple[float, int]] = []
        self.ok = device.type == "cuda"
        if self.ok:
            try:
                torch.cuda.utilization(device)
            except Exception:
                self.ok = False
        if self.ok:
            threading.Thread(target=self._run, args=(device,), daemon=True).start()

    def _run(self, device):
        while True:
            time.sleep(1.0)
            try:
                self.samples.append((time.time(), torch.cuda.utilization(device)))
            except Exception:
                return

    def mean(self, since: float) -> float | None:
        xs = [u for t, u in self.samples if t >= since]
        return sum(xs) / len(xs) if xs else None


def best_count(gains: list[float], threshold: float) -> int:
    """The n (1-based) maximizing gains[n - 1] - threshold * n."""
    return max(range(1, len(gains) + 1), key=lambda n: gains[n - 1] - threshold * n)


def fmt_e(x: float) -> str:
    return "nan" if math.isnan(x) else f"{x:.2e}"


__all__ = ["StepNet", "Group", "Stats", "root_group", "side_stats", "split_mse", "mse_now", "answers", "node_loss",
           "epoch_batches", "descend", "next_hyperplane", "top_direction", "Utilization", "best_count", "fmt_e",
           "EPS"]
