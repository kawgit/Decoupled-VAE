"""Refining delta trees: a code is the sequence of cells an image falls in, one per depth, each cell one of V
centroids that the encoder found at the image's node by running Lloyd's algorithm itself.

Remainders are as in dvae/deltatree.py: an image minus its node's predicted mean, the training mean plus the
deltas of the children taken so far. At a node the encoder holds V centroids, as offsets from the node's
predicted mean. Each network call is one Lloyd iteration: it reads the node's history and the current centroids
(one token per centroid, the tokens attending to each other and to the history) and predicts the next centroids
directly, as offsets from the node's predicted mean. Trained on the actual means of the cells the current
centroids make (each image to its nearest centroid), it learns to stand in for those means. A centroid token
reads its offset as a unit direction and log length (centroid_in="unit"), or as the raw offset and log length
("raw"); "pairs" adds a token per pair of centroids holding the hyperplane between their cells (its unit normal
and its distance from the node's predicted mean), which for V = 2 is dvae/deltatree.py's query. With
query="single" there is one query token instead, as in dvae/deltatree.py: the sum over pairs of centroids of
their hyperplane's embedding (plus one per pair), read by one head that gives all V centroids, and no step
embedding; for V = 2 its input is dvae/deltatree.py's query, and only its output (two free centroids, not r and
-m r) and the history (tokens, not sides and m) differ. tower=True (V = 2 only, an ablation) gives it
dvae/deltatree.py's output too: r and m, the children r and -m r.

Step 0's centroids come from --init: "pca", every node of a depth starting from the same offsets, plus and minus
each of the depth's top V/2 principal directions scaled by its rms (for V = 2 the same hyperplane as
dvae/deltatree.py's start); or "learned", a first call that reads only the history and predicts the centroids
from nothing (trained on the k-means objective of the cells it makes). With T steps, the last step's input
centroids make the node's partition (each image takes the token of its nearest one) and its output centroids,
their predicted means, are the children's: child v's delta is its centroid's offset.

The encoder predicts no shares; pricing its tokens is the prior's job (train_prior.py). As a diagnostic and the
prior's baseline, a node's implied shares are the weights w on the simplex for which the weighted mean of the
children's centroids is nearest the node's own predicted mean, |sum_v w_v offset_v|^2 least, floored at
SHARE_FLOOR: when the centroids are the cells' actual means and the node's mean is theirs, these are the cells'
shares exactly. Nothing the encoder, decoder or prior reads depends on them. Every query depends only on the
code, so decoding replays the encoder's choices.
"""

import contextlib
import itertools
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from dvae.deltatree import D, EPS, Block, ImageHead, MeanImage

SHARE_FLOOR = 1e-3
CENTROID_INPUTS = ("unit", "raw", "pairs")
QUERIES = ("tokens", "single")


@dataclass
class Frontier:
    """The nodes at one depth that hold images, and those images. The nodes' fields live on the CPU, or for a
    resident frontier (a batch's) on the images' device."""
    R: Tensor       # [N, D] each image's remainder
    node: Tensor    # [N] its node
    deltas: Tensor  # [Nn, k, D] half precision: each node's history, the delta of each child taken
    tokens: Tensor  # [Nn, k] long: the child taken at each depth
    kids: Tensor | None = None  # [Nn, k, V, D] half precision: every child's offset at each depth (once split)

    @property
    def depth(self):
        return self.tokens.shape[1]

    @property
    def nodes(self):
        return self.tokens.shape[0]

    def history(self, nodes, device):
        """History.memory's inputs for nodes: deltas, tokens and every child's offset (None before a split)."""
        nodes = nodes.to(self.deltas.device)
        kids = None if self.kids is None else self.kids[nodes].to(device).float()
        return self.deltas[nodes].to(device).float(), self.tokens[nodes].to(device), kids


def root_frontier(X, mean, resident=False):
    """Every image at the root; resident keeps the nodes' fields on the images' device."""
    store = X.device if resident else torch.device("cpu")
    return Frontier(R=X - mean, node=torch.zeros(len(X), dtype=torch.long, device=X.device),
                    deltas=torch.zeros(1, 0, X.shape[1], dtype=torch.half, device=store),
                    tokens=torch.zeros(1, 0, dtype=torch.long, device=store))


@dataclass
class Split:
    """Every node's split: the centroids that partition its images (part [Nn, V, D], offsets from its predicted
    mean), the children's centroids (cent [Nn, V, D], the predicted means of those cells) and their implied shares
    (w [Nn, V])."""
    part: Tensor
    cent: Tensor
    w: Tensor


def nearest(R: Tensor, offsets: Tensor, node: Tensor, chunk=8192) -> Tensor:
    """Each image's nearest centroid [N] (long), images R [N, D] at nodes node [N] with centroids offsets
    [Nn, V, D]."""
    out = []
    for i in range(0, len(R), chunk):
        r, c = R[i:i + chunk], offsets[node[i:i + chunk]]
        out.append(((c * c).sum(-1) - 2 * (c @ r[:, :, None])[..., 0]).argmin(-1))
    return torch.cat(out) if out else node.new_zeros(0)


_SUBSETS: dict[int, list[list[int]]] = {}
_INDICES: dict[tuple, Tensor] = {}


def _index(values: list[int], device) -> Tensor:
    """values as a long tensor on device, made once (a fresh one would be a host-to-device copy, which waits)."""
    key = (tuple(values), str(device))
    if key not in _INDICES:
        _INDICES[key] = torch.tensor(values, device=device)
    return _INDICES[key]


@torch.no_grad()
def shares(offsets: Tensor) -> Tensor:
    """The weights w [B, V] on the simplex that make sum_v w_v offsets_v [B, V, D] nearest zero (the node's
    predicted mean), floored at SHARE_FLOOR and renormalized. Exact: the best of every support's
    equality-constrained solution that is nonnegative (2^V - 1 supports)."""
    B, V, _ = offsets.shape
    o = offsets.double()  # (in double from the start: collinear centroids make G singular)
    G = o @ o.transpose(1, 2)  # [B, V, V]
    subsets = _SUBSETS.setdefault(V, [list(s) for n in range(1, V + 1) for s in itertools.combinations(range(V), n)])
    best_obj = G.new_full((B,), float("inf"))
    best_w = G.new_zeros(B, V)
    scale = G.diagonal(dim1=1, dim2=2).mean(-1).clamp_min(1e-30)  # (a ridge relative to the offsets' size)
    for s in subsets:
        idx = _index(s, G.device)
        Gs = G[:, idx][:, :, idx] + 1e-9 * scale[:, None, None] * torch.eye(len(s), dtype=G.dtype, device=G.device)
        x = torch.linalg.solve_ex(Gs, G.new_ones(B, len(s), 1))[0][..., 0]  # (solve would wait to check)
        total = x.sum(-1, keepdim=True)
        ws = x / torch.where(total.abs() > 0, total, torch.ones_like(total))
        ok = (ws >= -1e-9).all(-1) & (total[:, 0] > 0)
        w = G.new_zeros(B, V)
        w[:, idx] = ws.clamp_min(0)
        obj = (w[:, None, :] @ G @ w[:, :, None])[:, 0, 0]
        better = ok & (obj < best_obj - 1e-12 * scale)
        best_obj = torch.where(better, obj, best_obj)
        best_w = torch.where(better[:, None], w, best_w)
    w = best_w.float().clamp_min(SHARE_FLOOR)
    return w / w.sum(-1, keepdim=True)


class History(nn.Module):
    """A node's memory: its history, one token per child taken (its delta's direction and log length, and its
    token), with self-attention among them. A query reads it and itself at every layer (Block
    with mem), so a node's memory is computed once for all its queries. With siblings, each history token also
    reads every child's offset at that depth (direction and log length, plus a slot embedding, summed over the
    children), what dvae/deltatree.py's side and m tell about the other side, for any V. With node_mean, the
    memory also holds a token for the node's predicted mean image (the training mean, the buffer mean, plus the
    deltas)."""

    mean: Tensor  # [D] the training mean, with node_mean

    def __init__(self, depth, V, width, layers, heads, node_mean=False, siblings=False, pixels=D):
        super().__init__()
        self.width, self.D = width, pixels
        self.delta_in = nn.Linear(pixels + 1, width)
        self.siblings = siblings
        if siblings:
            self.kid_in = nn.Linear(pixels + 1, width)
            self.kid_slot = nn.Parameter(torch.randn(V, width) * 0.02)
        self.token_in = nn.Embedding(V, width)
        self.position = nn.Parameter(torch.randn(max(1, depth), width) * 0.02)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.mean_in = MeanImage(width) if node_mean else None
        self.extra = int(node_mean)
        if node_mean:
            assert pixels == D, "node_mean reads 28 x 28 images"
            self.mean_token = nn.Parameter(torch.randn(width) * 0.02)
            self.register_buffer("mean", torch.zeros(D))

    def memory(self, deltas, tokens, kids=None):
        """Each layer's normalized memory tokens [B, extra + k, W] for histories deltas [B, k, D] and tokens
        [B, k], and with siblings every child's offset kids [B, k, V, D]."""
        B, k = tokens.shape
        length = deltas.norm(dim=-1, keepdim=True).clamp_min(EPS)
        h = self.delta_in(torch.cat([deltas / length, length.log()], -1))
        h = h + self.token_in(tokens) + self.position[:k]
        if self.siblings and k > 0:
            assert kids is not None, "a siblings history needs every child's offset"
            size = kids.norm(dim=-1, keepdim=True).clamp_min(EPS)
            h = h + (self.kid_in(torch.cat([kids / size, size.log()], -1)) + self.kid_slot).sum(2)
        if self.mean_in is not None:
            h = torch.cat([(self.mean_in(self.mean + deltas.sum(1)) + self.mean_token)[:, None], h], 1)
        if h.shape[1] == 0:
            return [deltas.new_zeros(B, 0, self.width) for _ in self.blocks]
        mems = []
        for blk in self.blocks:
            h, x = blk(h)
            mems.append(x)
        return mems


def _position_rows(width, first, last):
    """History positions first..last - 1 [last - first, W], row i drawn from seed i (so a row is the same
    whenever it is added)."""
    rows = [torch.randn(width, generator=torch.Generator().manual_seed(i)) for i in range(first, last)]
    return (torch.stack(rows) if rows else torch.zeros(0, width)) * 0.02


class Refiner(History):
    """The encoder: one network for every depth below depths, each call one Lloyd iteration on a node's V
    centroids (module docstring). inits [depths, V, D] holds each depth's step-0 offsets for init="pca" (set by
    the trainer). grow adds depths."""

    inits: Tensor
    stops: Tensor  # [depths] long: the calls each depth's split uses (all of them unless set, dynamic steps)
    pair_index: Tensor  # [2, P] each pair's centroids, with pairs

    def __init__(self, depths, V=2, steps=16, init="pca", width=256, layers=4, heads=4, node_mean=False,
                 centroid_in="unit", query="tokens", tower=False, siblings=False, shape=(1, 28, 28)):
        shape = tuple(shape)
        super().__init__(depths, V, width, layers, heads, node_mean, siblings, shape[0] * shape[1] * shape[2])
        assert init in ("pca", "learned"), init
        assert centroid_in in CENTROID_INPUTS, centroid_in
        assert query in QUERIES, query
        assert query == "tokens" or centroid_in == "unit", "--centroid-in is for --query tokens"
        assert not tower or (query == "single" and V == 2), "--tower is for --query single with V = 2"
        assert init == "learned" or V % 2 == 0, "--init pca needs an even V (plus and minus each direction)"
        assert steps >= 1 or init == "learned", "--init pca needs at least one step"
        self.config = dict(depths=depths, V=V, steps=steps, init=init, width=width, layers=layers, heads=heads,
                           node_mean=node_mean, centroid_in=centroid_in, query=query, tower=tower,
                           siblings=siblings, shape=shape)
        self.V, self.steps, self.init, self.centroid_in, self.query = V, steps, init, centroid_in, query
        single = query == "single"
        self.pairs = list(itertools.combinations(range(V), 2)) if single or centroid_in == "pairs" else []
        if self.pairs:  # the hyperplane between each pair of centroids' cells
            self.register_buffer("pair_index", torch.tensor(self.pairs).T.contiguous(), persistent=False)
            self.pair_in = nn.Linear(self.D + 1, width)
            self.pair_slot = nn.Parameter(torch.randn(len(self.pairs), width) * 0.02)
        with torch.no_grad():
            self.position.copy_(_position_rows(width, 0, len(self.position)))
        if not single:
            self.offset_in = nn.Linear(self.D + 1, width)
            self.slot = nn.Parameter(torch.randn(V, width) * 0.02)
            self.step_in = nn.Parameter(torch.randn(steps + 1, width) * 0.02)  # row 0: the learned init's call
        self.blank = nn.Parameter(torch.randn(width) * 0.02)  # a query before there are centroids
        self.out_norm = nn.LayerNorm(width)
        self.tower = tower
        self.centroid_head = ImageHead(width, 1 if tower or not single else V, shape=shape)
        if tower:  # dvae/deltatree.py's output: r and m, the children r and -m r
            self.m_head = nn.Linear(width, 1)
        self.register_buffer("inits", torch.zeros(depths, V, self.D))
        self.register_buffer("stops", torch.full((depths,), self.calls, dtype=torch.long))
        self._stop_list: list[int] | None = None
        self.speed()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if prefix + "stops" not in state_dict:  # (from before dynamic steps: every depth uses every call)
            depths = state_dict[prefix + "inits"].shape[0]
            state_dict[prefix + "stops"] = torch.full((depths,), self.calls, dtype=torch.long)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._stop_list = None

    @property
    def depths(self):
        return self.inits.shape[0]

    @torch.no_grad()
    def grow(self, depths, inits):
        """Extend to depths, the new depths' step-0 offsets inits [depths - self.depths, V, D]. The position
        parameter keeps its identity (an optimizer's state for it must grow to match) and loses its gradient."""
        p = self.position
        p.data = torch.cat([p.data, _position_rows(self.width, len(p), max(1, depths)).to(p)])
        p.grad = None
        self.inits = torch.cat([self.inits, inits.to(self.inits)])
        self.stops = torch.cat([self.stops, self.stops.new_full((depths - len(self.stops),), self.calls)])
        self._stop_list = None
        self.config["depths"] = depths
        self.reset_graphs()

    def call(self, mems, offsets: Tensor | None, t: int | Tensor, B: int) -> Tensor:
        """One call at B nodes with memory mems: the centroids after it [B, V, D] from offsets [B, V, D] (None:
        the learned init's call, from nothing)."""
        V = self.V
        if self.query == "single":
            return self.single_call(mems, offsets, B)
        if offsets is None:
            h = (self.blank + self.slot).expand(B, -1, -1)
            if self.pairs:
                h = torch.cat([h, (self.blank + self.pair_slot).expand(B, -1, -1)], 1)
        else:
            length = offsets.norm(dim=-1, keepdim=True).clamp_min(EPS)
            if self.centroid_in == "raw":
                h = self.offset_in(torch.cat([offsets, length.log()], -1)) + self.slot
            else:
                h = self.offset_in(torch.cat([offsets / length, length.log()], -1)) + self.slot
            if self.pairs:
                h = torch.cat([h, self.pair_in(self.hyperplanes(offsets)) + self.pair_slot], 1)
        h = h + (self.step_in[t] if isinstance(t, int) else self.step_in[t][:, None])
        for blk, mem in zip(self.blocks, mems):
            h, _ = blk(h, mem)
        return self.centroid_head(self.out_norm(h[:, :V]))[:, :, 0]

    def hyperplanes(self, offsets: Tensor) -> Tensor:
        """Each pair (i, j)'s hyperplane between the cells of centroids offsets [B, V, D]: its unit normal and
        bias [B, P, D + 1], x being nearer c_i than c_j where x . normal >= bias."""
        i, j = self.pair_index[0], self.pair_index[1]
        diff = offsets[:, i] - offsets[:, j]  # [B, P, D]
        gap = diff.norm(dim=-1, keepdim=True).clamp_min(EPS)
        sq = offsets.square().sum(-1, keepdim=True)
        return torch.cat([diff / gap, (sq[:, i] - sq[:, j]) / (2 * gap)], -1)

    def single_call(self, mems, offsets: Tensor | None, B: int) -> Tensor:
        """query="single": one query token, the sum over pairs of centroids of their hyperplane's embedding
        (for V = 2 dvae/deltatree.py's query), reading the memory; one head gives all V centroids."""
        if offsets is None:
            q = self.blank.expand(B, 1, -1)
        else:
            q = (self.pair_in(self.hyperplanes(offsets)) + self.pair_slot).sum(1, keepdim=True)
        for blk, mem in zip(self.blocks, mems):
            q, _ = blk(q, mem)
        f = self.out_norm(q[:, 0])
        if self.tower:
            r, m = self.centroid_head(f)[:, 0], torch.sigmoid(self.m_head(f))
            return torch.stack([r, -m * r], 1)
        return self.centroid_head(f)

    @property
    def calls(self):
        """Network calls per node: the steps, plus the learned init's."""
        return self.steps + (self.init == "learned")

    def path(self, mems, B: int, depth: int, grad: set[int] | None = None, n: int | None = None):
        """Yields (partition, prediction) at B nodes of depth depth, for each of its first n calls (default all):
        the centroids whose cells the call's output is trained on (detached) and its output, the cells' predicted
        means [B, V, D]. The last pair is the node's Split (part, cent). With grad, only the calls it names
        (counted from 0) keep a graph for the backward pass; the rest run without one (each call's input is
        detached, so no gradient crosses from one call to the next)."""
        n = self.calls if n is None else n
        def run(i, offsets, t):
            if grad is None or i in grad:
                return self.run(mems, offsets, t, B)
            with torch.no_grad():
                return self.run(mems, offsets, t, B)

        i = 0
        if self.init == "learned":
            offsets = run(i, None, 0)
            i += 1
            yield offsets.detach(), offsets
        else:
            offsets = self.inits[depth].expand(B, -1, -1)
        for t in range(1, self.steps + 1):
            if i == n:
                return
            nxt = run(i, offsets.detach(), t)
            i += 1
            yield offsets.detach(), nxt
            offsets = nxt

    def speed(self, recompute=False, bf16=False, channels_last=False, graphs=False):
        """How calls run, none of it changing what they compute (beyond bf16's rounding): recompute keeps no call's
        activations for the backward pass but runs the call again there (memory for more nodes per piece);
        bf16 runs the network under bfloat16 autocast (centroids, partitions and losses stay float32);
        channels_last runs the centroid head's convolutions in that layout; graphs routes without autograd, each
        call a CUDA graph's replay (see routing), and trains on the chosen calls in one batch (chunk_sse)."""
        self.recompute, self.bf16, self.graphs = recompute, bf16, graphs
        self.centroid_head.channels_last = channels_last
        self.centroid_head.up.to(memory_format=torch.channels_last if channels_last else torch.contiguous_format)
        self.reset_graphs()

    def reset_graphs(self):
        """Forget the captured CUDA graphs (their memory lengths are the encoder's depths, which grow)."""
        self._graphs: dict[tuple, tuple] = {}
        self._graph_inputs: dict[tuple, tuple] = {}
        self._pool = None

    def amp(self):
        """The network's autocast context (bfloat16 with speed(bf16=True), else nothing). Its cast cache is off,
        as a CUDA graph's capture needs (casts then run each time, reading the weights as they are now)."""
        if not self.bf16:
            return contextlib.nullcontext()
        return torch.autocast(self.inits.device.type, dtype=torch.bfloat16, cache_enabled=False)

    def run(self, mems, offsets: Tensor | None, t: int | Tensor, B: int) -> Tensor:
        """call, as speed set it up; its centroids in float32."""
        with self.amp():
            if self.recompute and torch.is_grad_enabled():
                out = checkpoint(self.call, mems, offsets, t, B, use_reentrant=False)
            else:
                out = self.call(mems, offsets, t, B)
        assert isinstance(out, Tensor)
        return out.float()

    @torch.no_grad()
    def quick_call(self, mems, offsets: Tensor | None, t: int, B: int) -> Tensor:
        """call without autograd, its centroids in float32. With speed(graphs=True) on CUDA, a replay of a CUDA graph
        captured for this memory length (and step, for query="tokens"), at B padded up to a power of two: one
        launch instead of a call's hundred kernels. Padding rows hold stale values, which no real row reads."""
        if not (self.graphs and offsets is not None and offsets.is_cuda):
            return self.run(mems, offsets, t, B)
        Bb, k = 1 << max(B - 1, 0).bit_length(), mems[0].shape[1]
        key = (k, Bb, t if self.query == "tokens" else 0)
        if key not in self._graphs:
            self._capture(key, mems[0].dtype, t)
        graph, static_mems, static_off, out = self._graphs[key]
        for static, m in zip(static_mems, mems):
            static[:B].copy_(m)
        static_off[:B].copy_(offsets)
        graph.replay()
        return out[:B].clone()

    def _capture(self, key, dtype, t):
        k, Bb, _ = key
        device = self.inits.device
        if (Bb, dtype) not in self._graph_inputs:  # inputs shared by every memory length at this size
            K = self.depths + self.extra
            self._graph_inputs[(Bb, dtype)] = ([torch.zeros(Bb, K, self.width, dtype=dtype, device=device)
                                           for _ in self.blocks], torch.zeros(Bb, self.V, self.D, device=device))
        mems_buf, static_off = self._graph_inputs[(Bb, dtype)]
        static_mems = [m[:, :k] for m in mems_buf]
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):  # warm up (kernel choices, workspaces) outside the capture
            for _ in range(2):
                self.run(static_mems, static_off, t, Bb)
        torch.cuda.current_stream(device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            out = self.run(static_mems, static_off, t, Bb)
        self._pool = graph.pool()
        self._graphs[key] = (graph, static_mems, static_off, out)

    @torch.no_grad()
    def routing(self, mems, B: int, depth: int, n: int | None = None):
        """path without autograd (each call a quick_call): yields (partition, centroids) for each of the first n
        calls (default all)."""
        n = self.calls if n is None else n
        i = 0
        if self.init == "learned":
            offsets = self.quick_call(mems, None, 0, B)
            i += 1
            yield offsets, offsets
        else:
            offsets = self.inits[depth].expand(B, -1, -1)
        for t in range(1, self.steps + 1):
            if i == n:
                return
            nxt = self.quick_call(mems, offsets, t, B)
            i += 1
            yield offsets, nxt
            offsets = nxt

    def stop_at(self, depth: int) -> int:
        """The calls depth's split uses (from a copy on the host: reading the buffer would wait on the GPU)."""
        if self._stop_list is None:
            self._stop_list = self.stops.tolist()
        return self._stop_list[depth]

    def set_stops(self, stops: list[int]):
        """Each depth's calls for its split (dynamic steps)."""
        self.stops.copy_(torch.tensor(stops))
        self._stop_list = list(stops)

    def batched_calls(self, mems, calls: list[tuple[int, Tensor | None]], B: int) -> list[Tensor]:
        """Several calls at the same B nodes with memory mems, with autograd, as one batch: calls[i] = (its step,
        the offsets it starts from), None for the learned init's call (run on its own). Returns each call's
        centroids [B, V, D]."""
        out: list[Tensor | None] = [None] * len(calls)
        rest = [(i, t, x) for i, (t, x) in enumerate(calls) if x is not None]
        for i, (t, x) in enumerate(calls):
            if x is None:
                out[i] = self.run(mems, None, t, B)
        if rest:
            n = len(rest)
            steps = torch.tensor([t for _, t, _ in rest], device=self.inits.device).repeat_interleave(B)
            big = self.run([m.repeat(n, 1, 1) for m in mems], torch.cat([x for _, _, x in rest]), steps, n * B)
            for (i, _, _), y in zip(rest, big.split(B)):
                out[i] = y
        return [y for y in out if y is not None]


def cell_sse(pred: Tensor, R: Tensor, cell: Tensor, V: int) -> Tensor:
    """sum over cells of n |predicted mean - actual mean remainder|^2 [B]: pred [B, V, D] the cells' predicted
    means, images R [n, D] in cells cell [n] (node * V + token, long). An empty cell adds 0."""
    B = pred.shape[0]
    count = R.new_zeros(B * V).index_add_(0, cell, R.new_ones(len(R)))
    total = R.new_zeros(B * V, R.shape[1]).index_add_(0, cell, R)
    p = pred.reshape(B * V, R.shape[1])
    per = count * p.square().sum(-1) - 2 * (p * total).sum(-1) + total.square().sum(-1) / count.clamp_min(1)
    return per.view(B, V).sum(-1)


def chunk_sse(encoder: Refiner, fr: Frontier, imgs: Tensor | None, grad_calls=0, full=True):
    """The objective's sum over images imgs of a frontier (None: all of it): each node's cell error averaged over
    its calls, so every image counts once per depth. With 0 < grad_calls < the encoder's calls, the loss to
    backpropagate averages only that many calls drawn at random (the same expectation, a cheaper backward pass),
    while the objective returned for logging still averages them all. full runs every call; otherwise the depth
    stops after the calls its split uses (Refiner.stops). Returns the loss, the objective (detached), the
    chunk's nodes [b] and their Split (detached)."""
    if imgs is None:
        R, inverse = fr.R, fr.node
        nodes = torch.arange(fr.nodes, device=R.device)
    else:
        R = fr.R[imgs]
        nodes, inverse = torch.unique(fr.node[imgs], return_inverse=True)
    B, V = len(nodes), encoder.V
    history = fr.history(nodes, R.device)
    T = encoder.calls if full else encoder.stop_at(fr.depth)
    grad = set(torch.randperm(T)[:grad_calls].tolist()) if 0 < grad_calls < T else None
    if encoder.graphs:
        return _two_phase_sse(encoder, history, R, inverse, nodes, fr.depth, grad, T)
    with encoder.amp():
        mems = encoder.memory(*history)
    loss, objective = R.new_zeros(()), R.new_zeros(())
    part = cent = R.new_zeros(B, V, R.shape[1])
    for i, (part, cent) in enumerate(encoder.path(mems, B, fr.depth, grad, T)):
        cell = inverse * V + nearest(R, part, inverse)
        sse = cell_sse(cent, R, cell, V).sum()
        objective = objective + sse.detach()
        if grad is None or i in grad:
            loss = loss + sse
    cent = cent.detach()
    return loss / (T if grad is None else len(grad)), objective / T, nodes, Split(part, cent, shares(cent))


def _two_phase_sse(encoder: Refiner, history, R: Tensor, inverse: Tensor, nodes: Tensor, depth: int,
                   grad: set[int] | None, n: int):
    """chunk_sse with speed(graphs=True): every call runs first without autograd (encoder.routing), recording the
    partitions of the calls to train (grad, or all); then those calls run again together, with autograd, from the
    same inputs. Each call's input is detached, so this is the same loss and gradient, in far fewer kernel
    launches. The objective returned is the trained calls' (an unbiased estimate of all of them)."""
    B, V = len(nodes), encoder.V
    learned = encoder.init == "learned"
    with torch.no_grad(), encoder.amp():
        mems = encoder.memory(*history)
    chosen: list[tuple[int, Tensor | None]] = []
    cells = []
    part = cent = R.new_zeros(B, V, R.shape[1])
    for i, (part, cent) in enumerate(encoder.routing(mems, B, depth, n)):
        if grad is None or i in grad:  # (the other calls' cells are never needed: only their outputs, as inputs)
            chosen.append((i if learned else i + 1, None if learned and i == 0 else part))
            cells.append(inverse * V + nearest(R, part, inverse))
    with encoder.amp():
        mems = encoder.memory(*history)
    loss = sum((cell_sse(y, R, cell, V).sum() for y, cell in zip(encoder.batched_calls(mems, chosen, B), cells)),
               R.new_zeros(()))
    loss = loss / len(chosen)
    return loss, loss.detach(), nodes, Split(part, cent, shares(cent))


def cat_splits(parts: list[Split]) -> Split:
    return Split(torch.cat([p.part for p in parts]), torch.cat([p.cent for p in parts]),
                 torch.cat([p.w for p in parts]))


@torch.no_grad()
def choose(encoder: Refiner, fr: Frontier, batch: int, full=False) -> Split:
    """Every node's Split, batch nodes at a time: after the calls its depth uses (Refiner.stops), or with full
    after every call."""
    device = fr.R.device
    nodes = torch.arange(fr.nodes)
    parts = []
    for i in range(0, len(nodes), batch):
        part_nodes = nodes[i:i + batch]
        with encoder.amp():
            mems = encoder.memory(*fr.history(part_nodes, device))
        part = cent = None
        for part, cent in encoder.routing(mems, len(part_nodes), fr.depth,
                                          encoder.calls if full else encoder.stop_at(fr.depth)):
            pass
        assert part is not None and cent is not None
        parts.append(Split(part, cent, shares(cent)))
    return cat_splits(parts)


@torch.no_grad()
def children(fr: Frontier, s: Split, V: int, batch=8192):
    """The next depth's frontier, every node split by s and empty children dropped; each image's token [N] and
    its cost in bits under the shares [N]."""
    tok = nearest(fr.R, s.part, fr.node, batch)
    bits = -s.w[fr.node, tok].log2()
    delta = s.cent[fr.node, tok]
    keys, node = torch.unique(fr.node * V + tok, return_inverse=True)
    parent, ctok = keys // V, keys % V
    store = fr.tokens.device
    p = parent.to(store)
    kids = fr.kids if fr.kids is not None else s.cent.new_zeros(fr.nodes, 0, V, fr.R.shape[1], dtype=torch.half,
                                                                    device=store)
    child = Frontier(R=fr.R - delta, node=node,
                     deltas=torch.cat([fr.deltas[p], s.cent[parent, ctok].half().to(store)[:, None]], 1),
                     tokens=torch.cat([fr.tokens[p], ctok.to(store)[:, None]], 1),
                     kids=torch.cat([kids[p], s.cent[parent].half().to(store)[:, None]], 1))
    return child, tok, bits


@torch.no_grad()
def route_path(encoder: Refiner, fr: Frontier, batch: int):
    """Every node's Split (after the calls its depth uses, Refiner.stops), and each image's squared error after
    the split each of all the calls gives [calls, N] (its remainder less its cell's predicted mean, cells from
    that call's input centroids), so every number of steps can be scored."""
    device, Nn = fr.R.device, fr.nodes
    calls, stop = encoder.calls, encoder.stop_at(fr.depth)
    errs = fr.R.new_zeros(calls, len(fr.R))
    parts = []
    for i in range(0, Nn, batch):
        nodes = torch.arange(i, min(i + batch, Nn), device=device)
        mine = (fr.node >= i) & (fr.node < i + len(nodes))
        R, node = fr.R[mine], fr.node[mine] - i
        with encoder.amp():
            mems = encoder.memory(*fr.history(nodes, device))
        used = None
        for t, (part, cent) in enumerate(encoder.routing(mems, len(nodes), fr.depth)):
            tok = nearest(R, part, node)
            errs[t, mine] = (R - cent[node, tok]).square().sum(-1)
            if t + 1 == stop:
                used = Split(part, cent, shares(cent))
        assert used is not None
        parts.append(used)
    return cat_splits(parts), errs


class TokenDecoder(History):
    """p(x | leaf) for a refining tree's codes: dvae/deltatree.py's Decoder with History's memory (each child's
    delta, token and share). 256 logits per pixel, the sum of a path reading the history with a learned query
    and a few convolutions over the leaf's predicted mean image."""

    def __init__(self, depth, V=2, width=256, layers=4, heads=4, channels=64):
        super().__init__(depth, V, width, layers, heads)
        self.config = dict(depth=depth, V=V, width=width, layers=layers, heads=heads, channels=channels)
        self.query_token = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.out_norm = nn.LayerNorm(width)
        self.levels_head = ImageHead(width, 256)
        self.mean_path = nn.Sequential(nn.Conv2d(1, channels, 3, padding=1), nn.GELU(),
                                       nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
                                       nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
                                       nn.Conv2d(channels, 256, 1))

    def forward(self, deltas, tokens, mu):
        """Logits [B, 256, D] for leaves with histories deltas [B, k, D], tokens [B, k] and predicted means
        mu [B, D]."""
        q = self.query_token.expand(len(mu), 1, -1)
        for blk, mem in zip(self.blocks, self.memory(deltas, tokens)):
            q, _ = blk(q, mem)
        logits = self.levels_head(self.out_norm(q))[:, 0]
        return logits + self.mean_path(mu.view(-1, 1, 28, 28)).flatten(2)


def load_refiner(ckpt, device) -> Refiner:
    """train_refiner.py's checkpoint as its Refiner, in eval mode."""
    encoder = Refiner(**ckpt["encoder"]["config"]).to(device)
    encoder.load_state_dict(ckpt["encoder"]["state_dict"])
    return encoder.eval()
