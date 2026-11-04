"""CUDA graphs for train_walk.py's step: each depth's forward and backward recorded once and replayed as one
launch, since launching the step's many small kernels one by one costs more than running them.

A graph needs fixed shapes, so a depth's nodes are padded up to a power of two; a padded node holds no images,
so its loss and gradient are exactly 0. Every graph is recorded for one history length (the depth) and padded
node count, with fixed input buffers the step copies its frontier into, and they share one memory pool, which is
safe as they always replay one at a time in depth order. Routing the images to the children stays outside. The
parameters' gradients must stay allocated (zero_grad(set_to_none=False)): each replay adds into them.
"""

import torch

from dvae.deltatree import D, Frontier, SharedDecoder, Split, bisector, split_sse


class GraphedDepths:
    """Graphed depth steps for full batches of batch images, Lloyd steps queries per node and no seeds or
    tolerance, the loss divided by denom before its backward."""

    def __init__(self, decoder: SharedDecoder, batch, steps, denom):
        assert decoder.dir_head is None, "seeded decoders are not graphed"
        self.decoder, self.batch, self.steps, self.denom = decoder, batch, steps, denom
        device = decoder.starts.device
        self.params = [p for p in decoder.parameters() if p.requires_grad]
        for p in self.params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        self.most = 1 << (batch - 1).bit_length()  # the largest padded node count
        K = decoder.starts.shape[0]
        self.R = torch.zeros(batch, D, device=device)
        self.node = torch.zeros(batch, dtype=torch.long, device=device)
        self.deltas = torch.zeros(self.most, K, D, dtype=torch.half, device=device)
        self.sides = torch.zeros(self.most, K, device=device)
        self.ms = torch.zeros(self.most, K, device=device)
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs = {}  # (depth, padded nodes) -> (graph, its outputs)

    def body(self, k, P):
        """One depth's objective for the images in self.R at nodes self.node (of P), its backward, and the last
        step's split, as chunk_sse computes them."""
        dec, R, node = self.decoder, self.R, self.node
        mems = dec.memory(self.deltas[:P, :k].float(), self.sides[:P, :k], self.ms[:P, :k])
        all_n = R.new_zeros(P).index_add_(0, node, R.new_ones(len(R)))
        all_sum = R.new_zeros(P, D).index_add_(0, node, R)
        n, b = dec.start_at(k).expand(P, D), R.new_zeros(P)
        total = R.new_zeros(P)
        last = (n, b, n, b)
        for step in range(self.steps):
            r, m, _ = dec.query(mems, n, b)
            rare = (R * n[node]).sum(-1) >= b[node]
            rare_n = R.new_zeros(P).index_add_(0, node, rare.to(R.dtype))
            rare_sum = R.new_zeros(P, D).index_add_(0, node, R * rare[:, None])
            total = total + split_sse(r, m, rare_n, rare_sum, all_n, all_sum)
            last = (n, b, r.detach(), m.detach())
            if step + 1 < self.steps:
                n, b = bisector(r.detach(), m.detach())
        sse = total.sum() / self.steps
        (sse / self.denom).backward()
        return sse.detach(), last

    def record(self, k, P):
        saved = [p.grad.clone() for p in self.params if p.grad is not None]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm up (cuDNN picks its algorithms) off the graph
            for _ in range(2):
                self.body(k, P)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out = self.body(k, P)
        for p, g in zip(self.params, saved):  # undo the warm-ups' gradients
            assert p.grad is not None
            p.grad.copy_(g)
        self.graphs[(k, P)] = (graph, out)

    def run(self, fr: Frontier):
        """chunk_sse's objective over all of fr's images (backward done) and every node's Split."""
        k, Nn = fr.depth, len(fr.bits)
        P = 1 << max(0, Nn - 1).bit_length()
        self.R.copy_(fr.R)
        self.node.copy_(fr.node)
        self.deltas[:Nn, :k].copy_(fr.deltas)
        self.sides[:Nn, :k].copy_(fr.sides)
        self.ms[:Nn, :k].copy_(fr.ms)
        if (k, P) not in self.graphs:
            self.record(k, P)
        graph, (sse, (n, b, r, m)) = self.graphs[(k, P)]
        graph.replay()
        steps = torch.full((Nn,), float(self.steps), device=n.device)
        return sse.clone(), Split(n[:Nn].clone(), b[:Nn].clone(), r[:Nn].clone(), m[:Nn].clone(), None, steps)
