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

import torch
import torch.nn as nn
from torch import Tensor

from dvae.components import POSITIONS, Group
from dvae.deltatree import EPS, Block, HistoryReader, ImageHead, splitmix64


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
def vanswers(net: VStepNet, g: Group, q: Tensor, batch=1024, out: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """The network's predicted means c [Nn, V, D] (in q's dtype) and shares p [Nn, V] at every node of g, asked
    about q. out: c's tensor (q itself may be: each batch of nodes is read before it is written)."""
    net.eval()
    c = out if out is not None else torch.empty(q.shape, dtype=q.dtype, device=q.device)
    p = torch.empty(q.shape[:2], device=q.device)
    for i in range(0, g.nodes, batch):
        nodes = torch.arange(i, min(i + batch, g.nodes), device=g.R.device)
        cn, p[nodes] = net.query(net.memory_q(*g.history_q(nodes)), q[nodes])
        c[nodes] = cn.to(c.dtype)
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
def vdescend(g: Group, cells: Tensor, c: Tensor, p: Tensor, gpu_budget: float, table_bits: int = 8):
    """Split every node of g into the cells (each image's [N]) with predicted means c [Nn, V, D] and shares p:
    g moves to the next depth. The history tables keep each child's delta, its token as its side and its share
    as its m (a plain history reads neither). Returns what the run keeps of the split."""
    from dvae.components import refresh_stats
    V = c.shape[1]
    children, node = torch.unique(g.node * V + cells, return_inverse=True)
    parent, token = children // V, children % V
    delta = c[parent, token].float()
    g.R = g.R - delta[node]
    device = g.R.device
    if table_bits == 8:
        scale = delta.abs().amax(-1).clamp_min(1e-12) / 127
        table, size = (delta / scale[:, None]).round().to(torch.int8), 1
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
        kept = table.float() * scale[:, None] if table_bits == 8 else table.float()
        early = delta - kept if g.early is None else g.early[parent] + (delta - kept)
        old = g.depth - g.window - 1
        if old >= 0:
            t_old, s_old, at = g.deltas[old], g.scales[old], g.anc[:, old]
            row = t_old[at.to(t_old.device)].to(device).float()
            early = early + (row if s_old is None else row * s_old[at, None])
            g.deltas[old] = t_old.new_zeros(0, delta.shape[1])
        g.early, g.sums = early, None  # (history_q reads early; sums is history's)
    g.keys = splitmix64(g.keys[parent.cpu()] * V + token.cpu())
    g.node = node
    refresh_stats(g)
    return dict(token=cells.to(torch.int8).cpu(), node=node.int().cpu(), parent=parent.int().cpu(),
                child=token.to(torch.int8).cpu(), share=p.half().cpu())


__all__ = ["VStepNet", "CellSums", "cells_of", "cell_stats", "vsplit_sse", "attraction_sse", "vsplit_mse", "vanswers",
           "pca_start", "vdescend"]
