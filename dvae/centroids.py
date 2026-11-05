"""A two-centroid encoder: at every node, the network predicts the node's two children's centroids directly,
instead of side means for a hyperplane that Lloyd's algorithm then moves.

Given a node's history (the same HistoryReader memory as the delta tree's Encoder), a learned query reads out two
images, the centroids c0 and c1, as deltas from the node's predicted mean in pixel units. An image goes to the
nearer centroid, which is the side of the hyperplane bisecting them, and that centroid is the child's delta. The
training loss at a node is each image's squared distance to its nearer centroid, the k-means objective with
k = 2: its gradient pulls each centroid toward the mean of the images it won, so training does Lloyd's algorithm
across batches, and the trained network gives the settled centroids in one pass.

There is no share m here: a history token holds the chosen centroid's delta, its side (+1 for c1, -1 for c0) and
0 in the m slot. Every query depends only on the code, so decoding replays the encoder's choices exactly.
"""

import torch
import torch.nn as nn
from torch import Tensor

from dvae.deltatree import D, Frontier, HistoryReader, ImageHead, child_key


def position_rows(width, first, last):
    """History positions first..last - 1 [last - first, width], row i drawn from seed i (as Encoder's are)."""
    rows = [torch.randn(width, generator=torch.Generator().manual_seed(i)) for i in range(first, last)]
    return (torch.stack(rows) if rows else torch.zeros(0, width)) * 0.02


class CentroidEncoder(HistoryReader):
    """One network for every depth below depths. grow adds depths: the only part sized by them is the history
    positions."""

    def __init__(self, depths, width=256, layers=4, heads=4, node_mean=False):
        super().__init__(depths, width, layers, heads, node_mean)
        self.config = dict(depths=depths, width=width, layers=layers, heads=heads, node_mean=node_mean)
        self.query_token = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.out_norm = nn.LayerNorm(width)
        self.head = ImageHead(width, 2)
        with torch.no_grad():
            self.position.copy_(position_rows(width, 0, len(self.position)))

    @property
    def depths(self) -> int:
        return self.config["depths"]

    @torch.no_grad()
    def grow(self, depths):
        """Extend to depths. The position parameter keeps its identity (an optimizer's state for it must grow to
        match) and loses its gradient."""
        p = self.position
        p.data = torch.cat([p.data, position_rows(self.width, len(p), max(1, depths)).to(p)])
        p.grad = None
        self.config["depths"] = depths

    def centroids(self, mems: list[Tensor]) -> Tensor:
        """The two centroids [B, 2, D] (deltas from each node's predicted mean) of B nodes with memory mems."""
        q = self.query_token.expand(len(mems[0]), 1, -1)
        for blk, mem in zip(self.blocks, mems):
            q, _ = blk(q, mem)
        return self.head(self.out_norm(q))[:, 0]


def nearer(R, C):
    """Each image's side [N] (True: c1 is nearer, ties to c0) and squared distance to its nearer centroid [N], for
    remainders R [N, D] and each image's node's centroids C [N, 2, D]."""
    d = (R[:, None] - C).square().sum(-1)
    return d[:, 1] < d[:, 0], d.min(1).values


@torch.no_grad()
def centroid_children(fr: Frontier, C: Tensor, batch=8192):
    """The next depth's frontier when every node splits by its centroids C [Nn, 2, D] (on the images' device),
    empty children dropped; and each image's side [N]."""
    side = torch.cat([nearer(fr.R[i:i + batch], C[fr.node[i:i + batch]])[0] for i in range(0, len(fr.R), batch)])
    children, node = torch.unique(fr.node * 2 + side, return_inverse=True)
    parent, one = children // 2, (children % 2).bool()
    child_delta = C[parent, one.long()]
    store = fr.bits.device
    p, one_s = parent.to(store), one.to(store)
    child = Frontier(R=fr.R - C[fr.node, side.long()], node=node,
                     bits=child_key(fr.bits[p], one_s, fr.depth),
                     deltas=torch.cat([fr.deltas[p], child_delta.half().to(store)[:, None]], 1),
                     sides=torch.cat([fr.sides[p], (2 * one_s.float() - 1)[:, None]], 1),
                     ms=torch.cat([fr.ms[p], torch.zeros(len(p), 1, device=store)], 1))
    return child, side
