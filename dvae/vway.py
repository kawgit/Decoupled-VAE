"""V-way splits for the frozen-components inner encoder (train_inner_vway.py): dvae/components.py with every node
split into V children instead of two.

A Lloyd step at a node is asked about V centroids (offsets from the node's predicted mean): they make the node's
partition, each image in the cell of its nearest centroid. The step's network reads the node's history as a
StepNet does (dvae/deltatree.py's HistoryReader) and the V centroids as V query tokens (each one's direction and
log length, plus an embedding of its slot), which attend to the history and to each other at every layer. Each
token's output gives that cell's predicted mean (the same image head as StepNet's r) and a share logit; the
predicted means are centered so that their share-weighted mean is the node's (the tower property; for V = 2 this
is r and -m r). Trained on each cell's exact count and sum (vsplit_sse: the excess of the split's error over the
cells' own scatter), the network learns to stand in for the cells' actual means, and the next step is asked about
its answers. For V = 2 the cells are the sides of the bisector, so this is dvae/components.py's step with a
different query embedding.

A depth's step 0 asks every node about the same centroids: plus and minus each of the training remainders' top
V / 2 principal directions there, scaled by their rms along it (for V = 2, the binary tree's start).

With attraction > 0 the loss also pulls every predicted mean toward all of its node's images (attraction times
the sum of their squared distances to it), so a centroid whose cell is empty still moves (dvae/rounds.py's on the
lloyd-refine branch).
"""

import contextlib

import torch
import torch.nn as nn
from torch import Tensor

from dvae.components import POSITIONS, Group
from dvae.deltatree import EPS, Block, HistoryReader, ImageHead, splitmix64
from dvae.node import NodeEncoder, QueryReader, directions


class VStepNet(HistoryReader):
    """One (depth, Lloyd step)'s network for V-way splits (see the module's docstring)."""

    def __init__(self, V=4, width=128, layers=4, heads=4, plain_history=True, shape=(1, 28, 28), dropout=0.0,
                 head="token", head_hidden=1):
        shape = tuple(shape)
        super().__init__(POSITIONS, width, layers, heads, False, plain_history, shape[0] * shape[1] * shape[2])
        assert head in ("token", "shared"), head
        self.config = dict(V=V, width=width, layers=layers, heads=heads, plain_history=plain_history, shape=shape,
                           dropout=dropout, head=head, head_hidden=head_hidden)
        self.V, self.shared = V, head == "shared"
        self.centroid_in = nn.Linear(self.D + 1, width)
        self.slot = nn.Parameter(torch.randn(V, width) * 0.02)
        self.out_norm = nn.LayerNorm(width)
        # (token: each centroid's mean from its own token, V calls of the image head, which on small images costs
        # more than the blocks; shared: one call reading all V tokens' features, V channels out)
        self.c_head = ImageHead(width * V, V, shape=shape, hidden=head_hidden) if self.shared \
            else ImageHead(width, 1, shape=shape, hidden=head_hidden)
        self.p_head = nn.Linear(width, 1)
        if dropout > 0:  # (as StepNet: attention weights, both branches' outputs)
            for blk in self.blocks:
                assert isinstance(blk, Block) and isinstance(blk.ff, nn.Sequential)
                blk.dropout = dropout
                blk.ff.append(nn.Dropout(dropout))

    def query(self, mems, q: Tensor) -> tuple[Tensor, Tensor]:
        """For V centroids q [B, V, D] at each of B nodes with memory mems: the cells' predicted means c [B, V, D]
        (their share-weighted mean 0) and shares p [B, V]."""
        q = q.float()  # (the run may keep its centroids in bfloat16)
        length = q.norm(dim=-1, keepdim=True).clamp_min(EPS)
        with self.amp():
            h = self.centroid_in(torch.cat([q / length, length.log()], -1)) + self.slot
            for blk, mem in zip(self.blocks, mems):
                h, _ = blk(h, mem)
            f = self.out_norm(h)
            o = self.c_head(f.flatten(1)) if self.shared else self.c_head(f)[:, :, 0]
            o, logits = o.float(), self.p_head(f)[..., 0].float()
        p = logits.softmax(-1)
        return o - (p[..., None] * o).sum(1, keepdim=True), p


class VNodeStepNet(nn.Module):
    """VStepNet's job (the same memory_q and query, so the same callers) with the node read by dvae/node.py's
    NodeEncoder: its current mean (the deltas taken, summed: the history's early token plus its window; the tree's
    mean is a constant, in the token's bias) and its last `window` deltas, the V centroids as queries reading it
    (QueryReader). Sides and ms are not read (VStepNet's plain history)."""

    def __init__(self, V=4, width=128, layers=4, heads=4, window=16, shape=(1, 28, 28), dropout=0.0, head="token",
                 head_hidden=1):
        super().__init__()
        shape = tuple(shape)
        assert head in ("token", "shared"), head
        self.config = dict(kind="node", V=V, width=width, layers=layers, heads=heads, window=window, shape=shape,
                           dropout=dropout, head=head, head_hidden=head_hidden)
        self.V, self.window, self.shared = V, window, head == "shared"
        self.D = shape[0] * shape[1] * shape[2]
        self.bf16 = False  # (set by the trainer, as HistoryReader's)
        self.fp16 = False
        self.node = NodeEncoder(0, self.D, window, width, layers, heads, dropout)
        self.reader = QueryReader(width, layers, heads, dropout)
        self.centroid_in = nn.Linear(self.D + 1, width)
        self.slot = nn.Parameter(torch.randn(V, width) * 0.02)
        self.c_head = ImageHead(width * V, V, shape=shape, hidden=head_hidden) if self.shared \
            else ImageHead(width, 1, shape=shape, hidden=head_hidden)
        self.p_head = nn.Linear(width, 1)

    def amp(self):
        """As HistoryReader.amp."""
        if self.fp16:
            return torch.autocast(self.slot.device.type, dtype=torch.float16)
        if not self.bf16:
            return contextlib.nullcontext()
        return torch.autocast(self.slot.device.type, dtype=torch.bfloat16)

    def memory_q(self, raw, scales, early, sides, ms) -> tuple[list[Tensor], Tensor]:
        """The node's memory and summary, from Group.history_q's window (int8 or float16 rows [B, k, D], scales
        [B, k]; k < window at shallow depths: the window is padded before its oldest) and early token."""
        B, k, D = raw.shape
        win = raw.float() * scales[..., None]
        mean = win.sum(1) if early is None else early.float() + win.sum(1)
        pad = self.window - k
        assert pad >= 0, f"a history of {k} deltas, more than the window's {self.window}"
        win = torch.cat([win.new_zeros(B, pad, D), win], 1)
        valid = torch.arange(self.window, device=raw.device).expand(B, -1) >= pad
        with self.amp():
            return self.node(win, valid, mean, torch.zeros(B, dtype=torch.long, device=raw.device))

    def query(self, memory, q: Tensor) -> tuple[Tensor, Tensor]:
        """As VStepNet.query, memory memory_q's (the node's memory and summary)."""
        mems, summary = memory
        with self.amp():
            h = self.centroid_in(torch.cat(directions(q.float()), -1)) + self.slot
            f = self.reader(h, mems, summary)
            o = self.c_head(f.flatten(1)) if self.shared else self.c_head(f)[:, :, 0]
            o, logits = o.float(), self.p_head(f)[..., 0].float()
        p = logits.softmax(-1)
        return o - (p[..., None] * o).sum(1, keepdim=True), p


StepNet = VStepNet | VNodeStepNet


def make_step_net(config: dict) -> StepNet:
    """A V-way step network from its config (a saved one's: VStepNet's have no kind)."""
    config = dict(config)
    kind = config.pop("kind", "vstep")
    return VNodeStepNet(**config) if kind == "node" else VStepNet(**config)


def rows_per_chunk(V: int, D: int, chunk: int) -> int:
    """At most chunk rows of [V, D] float32 at a time, and at most 512 MiB of them (large V and D)."""
    return max(64, min(chunk, 2**29 // (V * D * 4)))


def cells_of(g: Group, q: Tensor, chunk=8192) -> Tensor:
    """Each image's cell [N] (long): its nearest of its node's centroids q [Nn, V, D] (a chunk of images at a time,
    so nothing of q's size is made)."""
    chunk = rows_per_chunk(q.shape[1], q.shape[2], chunk)
    cells = []
    for i in range(0, len(g.R), chunk):
        qi = q[g.node[i:i + chunk]].float()
        cells.append((qi.square().sum(-1) - 2 * torch.einsum("nd,nvd->nv", g.R[i:i + chunk], qi)).argmin(-1))
    return torch.cat(cells)


class CellSums:
    """Each node's cells' remainder sums, as a [Nn, V, D] tensor reads, but held only for the cells some image is in
    (at most one per image, where the dense tensor has Nn V rows, mostly empty once most nodes hold an image or
    two): sums[nodes] (an index tensor or a slice of nodes) is that part of the dense tensor, an empty cell's
    row zeros."""

    def __init__(self, rows: Tensor, row: Tensor):
        self.rows = rows  # [K + 1, D]: the K non-empty cells' sums, then a row of zeros
        self.row = row  # [Nn, V]: each cell's row in rows (an empty cell's the zeros)

    def __getitem__(self, nodes) -> Tensor:
        return self.rows[self.row[nodes]]


def cell_stats(g: Group, q: Tensor, chunk=8192) -> tuple[Tensor, CellSums, Tensor]:
    """Each node's cells' image counts [Nn, V] and remainder sums (CellSums, [Nn, V, D] as read) under
    centroids q, and each image's cell [N] (int8)."""
    Nn, V, D = q.shape
    cells = cells_of(g, q, chunk)
    at = g.node * V + cells  # (each image's cell among all nodes')
    count = g.R.new_zeros(Nn * V).index_add_(0, at, g.R.new_ones(len(at)))
    cell, k = torch.unique(at, return_inverse=True)
    rows = g.R.new_zeros(len(cell) + 1, D)
    for i in range(0, len(g.R), chunk):
        rows.index_add_(0, k[i:i + chunk], g.R[i:i + chunk])
    row = torch.full((Nn * V,), len(cell), dtype=torch.long, device=g.R.device)
    row[cell] = torch.arange(len(cell), device=g.R.device)
    return count.view(Nn, V), CellSums(rows, row.view(Nn, V)), cells.to(torch.int8)


def vsplit_sse(c: Tensor, count: Tensor, total: Tensor) -> Tensor:
    """sum over cells of n |predicted mean - actual mean|^2 [B], from each cell's count [B, V] and sum [B, V, D]
    (an empty cell adds 0)."""
    return (count * c.square().sum(-1) - 2 * (c * total).sum(-1)
            + total.square().sum(-1) / count.clamp_min(1)).sum(-1)


def attraction_sse(c: Tensor, n: Tensor, total: Tensor) -> Tensor:
    """sum over each node's images x and its V predicted means c of |x - c|^2, less what does not depend on c [B]:
    from the node's count n [B] and remainder sum total [B, D]."""
    return (n[:, None] * c.square().sum(-1) - 2 * (c * total[:, None]).sum(-1)).sum(-1)


def vsplit_mse(g: Group, count: Tensor, total: CellSums, c: Tensor, chunk=8192) -> float:
    """The mse per number after splitting g's nodes into the cells (count, total) with predicted means c, exactly:
    each node's sum of squared remainders less sum over cells of (2 c . sum - n |c|^2) (a chunk of nodes at a time)."""
    assert g.stats is not None
    sse = g.stats.sq.sum()
    chunk = rows_per_chunk(c.shape[1], c.shape[2], chunk)
    for i in range(0, len(c), chunk):
        ci = c[i:i + chunk].float()
        sse = sse - (2 * (ci * total[i:i + chunk]).sum(-1) - count[i:i + chunk] * ci.square().sum(-1)).sum()
    return sse.item() / (len(g.R) * g.R.shape[1])


@torch.no_grad()
def vanswers(net: StepNet, g: Group, q: Tensor, batch=1024, out: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """The network's predicted means c [Nn, V, D] (in q's dtype) and shares p [Nn, V] at every node of g, asked
    about q. out: c's tensor (q itself may be: each batch of nodes is read before it is written)."""
    net.eval()
    c = out if out is not None else torch.empty(q.shape, dtype=q.dtype, device=q.device)
    p = torch.empty(q.shape[:2], device=q.device)
    for i in range(0, g.nodes, batch):
        n = min(batch, g.nodes - i)
        # (the last batch padded to a multiple of 64 with its last node: fewer shapes for compiled code and cuDNN)
        nodes = torch.arange(i, i + n + (-n % 64), device=g.R.device).clamp_max(g.nodes - 1)
        cn, pn = net.query(net.memory_q(*g.history_q(nodes)), q[nodes])
        c[i:i + n], p[i:i + n] = cn[:n].to(c.dtype), pn[:n]
    return c, p


@torch.no_grad()
def pca_start(R: Tensor, V: int, chunk=8192) -> Tensor:
    """A depth's step-0 centroids [V, D]: plus and minus each of the remainders R's top V / 2 principal directions
    (each signed so that its largest entry is positive), scaled by the rms along it."""
    assert V % 2 == 0, "V-way starts pair each principal direction with its opposite: V must be even"
    gram = sum(R[i:i + chunk].T @ R[i:i + chunk] for i in range(0, len(R), chunk))
    assert isinstance(gram, Tensor)
    gram = gram.double() if gram.device.type == "cuda" else gram.cpu().double()
    vals, vecs = torch.linalg.eigh(gram)
    out = []
    for k in range(V // 2):
        v = vecs[:, -1 - k]
        v = v * v[v.abs().argmax()].sign() * (vals[-1 - k].clamp_min(0) / len(R)).sqrt()
        out += [v, -v]
    return torch.stack(out).float().to(R.device)


@torch.no_grad()
def split_rows(g: Group, cells: Tensor, c: Tensor, p: Tensor, keep_c: bool = False) -> dict:
    """What vdescend needs of the answers c [Nn, V, D] and shares p [Nn, V] that split g's nodes into the cells
    (each image's [N]): the children (node * V + cell, the non-empty cells, in order), each image's child, each
    child's predicted mean [children, D] (in c's dtype) and p; and c itself (a copy) with keep_c. Taken while c
    is on the GPU, so that a step's answers need not be kept until the depth splits (at V = 16 on ImageNet
    latents they are 12 GiB, the children's means 0.8)."""
    V = c.shape[1]
    children, node = torch.unique(g.node * V + cells, return_inverse=True)
    return dict(V=V, children=children, node=node, delta=c[children // V, children % V], p=p,
                c=c.clone() if keep_c else None)


@torch.no_grad()
def vdescend(g: Group, cells: Tensor, rows: dict, gpu_budget: float, table_bits: int = 8):
    """Split every node of g into the cells (each image's [N]) with split_rows' children, their predicted means
    and the shares: g moves to the next depth. The history tables keep each child's delta, its token as its side
    and its share as its m (a plain history reads neither). Returns what the run keeps of the split."""
    from dvae.components import refresh_stats
    V, children, node, p = rows["V"], rows["children"], rows["node"], rows["p"]
    parent, token = children // V, children % V
    delta = rows["delta"].float()
    g.R.sub_(delta[node])  # (in place, as below: at V = 16 the GPU is nearly full when a depth splits)
    device = g.R.device
    if table_bits == 8:
        scale = delta.abs().amax(-1).clamp_min(1e-12) / 127
        table, size = (delta / scale[:, None]).round_().to(torch.int8), 1
        g.scales.append(scale)
    else:
        table, size = delta.half(), 2
        g.scales.append(None)
    on_gpu = device.type != "cuda" or torch.cuda.memory_allocated(device) + delta.numel() * size < gpu_budget
    g.deltas.append(table if on_gpu else table.cpu().pin_memory())
    g.sides.append(token.float())
    g.ms.append(p[parent, token].clone())
    g.anc = torch.cat([g.anc[parent], torch.arange(len(children), device=device)[:, None]], 1)
    if g.window > 0:
        # (early: each node's sum of deltas less its window's as the tables keep them, history_q's early token.
        # A child's is its parent's, plus what its own table entry rounds off, plus the delta leaving the window)
        kept = table.float().mul_(scale[:, None]) if table_bits == 8 else table.float()
        early = delta.sub_(kept)  # (delta's last use)
        del kept
        if g.early is not None:
            early = g.early[parent].add_(early)
        old = g.depth - g.window - 1
        if old >= 0:
            t_old, s_old, at = g.deltas[old], g.scales[old], g.anc[:, old]
            row = t_old[at.to(t_old.device)].to(device).float()
            early.add_(row if s_old is None else row.mul_(s_old[at, None]))
            del row
            g.deltas[old] = t_old.new_zeros(0, early.shape[1])
        g.early, g.sums = early, None  # (history_q reads early; sums is history's)
    g.keys = splitmix64(g.keys[parent.cpu()] * V + token.cpu())
    g.node = node
    refresh_stats(g)
    return dict(token=cells.to(torch.int8).cpu(), node=node.int().cpu(), parent=parent.int().cpu(),
                child=token.to(torch.int8).cpu(), share=p.half().cpu())


def _live(g: Group) -> range:
    """The tables g's window still reads (vdescend has emptied those before: their sides, shares and scales go
    unread too)."""
    return range(max(0, g.depth - g.window) if g.window > 0 else 0, g.depth)


def _like(x: Tensor, table: Tensor) -> Tensor:
    """x where table is: pinned host memory for a table vdescend put there."""
    return x.pin_memory() if table.device.type == "cpu" and table.is_pinned() else x


def _drop_unread(g: Group):
    """Empty the sides, shares and scales of the tables g's window no longer reads (vdescend keeps them)."""
    live = _live(g)
    for j in range(live.start):
        g.sides[j], g.ms[j] = g.sides[j].new_zeros(0), g.ms[j].new_zeros(0)
        s = g.scales[j]
        g.scales[j] = None if s is None else s.new_zeros(0)


@torch.no_grad()
def keep_images(g: Group, idx: Tensor):
    """Keep only g's images idx [n] (indices on g's device, in their new order), the nodes they are in and the
    table rows those read (a rolling buffer's oldest images leave so)."""
    from dvae.components import refresh_stats
    used, node = torch.unique(g.node[idx], return_inverse=True)
    g.R, g.node = g.R[idx], node
    g.keys = g.keys[used.cpu()]
    g.early = None if g.early is None else g.early[used]
    g.sums = None if g.sums is None else g.sums[used]
    anc = g.anc[used]
    _drop_unread(g)
    for j in _live(g):
        rows, at = torch.unique(anc[:, j], return_inverse=True)
        anc[:, j] = at
        table = g.deltas[j]
        g.deltas[j] = _like(table[rows.to(table.device)], table)
        s = g.scales[j]
        g.scales[j] = None if s is None else s[rows]
        g.sides[j], g.ms[j] = g.sides[j][rows], g.ms[j][rows]
    g.anc = anc
    refresh_stats(g)


@torch.no_grad()
def append_group(g: Group, h: Group):
    """Add h's images after g's: h is a group of other images at g's depth with g's window and table bits (coded
    by the same networks), its nodes going after g's and its table rows after g's. Nodes with the same code in
    both stay apart: as two nodes they get the same answers and the same loss as one."""
    from dvae.components import refresh_stats
    assert g.depth == h.depth and g.window == h.window, (g.depth, h.depth, g.window, h.window)
    assert (g.early is None) == (h.early is None) and (g.sums is None) == (h.sums is None)
    _drop_unread(g)
    anc = h.anc.clone()
    anc[:, :_live(g).start] = 0  # (rows of emptied tables: never read)
    for j in _live(g):
        table = g.deltas[j]
        anc[:, j] += len(table)
        g.deltas[j] = _like(torch.cat([table, h.deltas[j].to(table.device)]), table)
        s, sh = g.scales[j], h.scales[j]
        g.scales[j] = None if s is None or sh is None else torch.cat([s, sh])
        g.sides[j], g.ms[j] = torch.cat([g.sides[j], h.sides[j]]), torch.cat([g.ms[j], h.ms[j]])
    g.node = torch.cat([g.node, h.node + g.nodes])
    g.keys = torch.cat([g.keys, h.keys])
    g.R, g.anc = torch.cat([g.R, h.R]), torch.cat([g.anc, anc])
    if g.early is not None and h.early is not None:
        g.early = torch.cat([g.early, h.early])
    if g.sums is not None and h.sums is not None:
        g.sums = torch.cat([g.sums, h.sums])
    refresh_stats(g)


__all__ = ["VStepNet", "VNodeStepNet", "StepNet", "make_step_net", "CellSums", "split_rows", "cells_of", "cell_stats",
           "vsplit_sse", "attraction_sse", "vsplit_mse", "vanswers", "pca_start", "vdescend", "keep_images",
           "append_group"]
