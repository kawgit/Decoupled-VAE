"""A train_inner_vway.py run's tree, replayed on any images: each image coded through the run's saved networks
(models/dDDD_sSS.pt, the kept steps of every depth as records.json gives them) exactly as the run codes test, so
the codes and remainders are the run's own (up to the arithmetic of other batch shapes and, on another GPU, another
precision). What the outer decoder's fine-tuning (finetune_outer_decoder.py) and a V-way prior read.
"""

import json
import os

import torch
from torch import Tensor

from dvae.components import Group, root_group
from dvae.vway import cells_of, make_step_net, split_rows, vanswers, vdescend


class VTree:
    def __init__(self, run: str, device: torch.device, depths: int | None = None, fp16: bool = False,
                 node_batch: int = 2048, gpu_tables: float = 0.75):
        """run's first depths depths (default: every depth its records.json has; the run may still be going, and
        a depth's networks are saved before its record). The networks are held in host memory in float16 (the
        run's own computed in bfloat16 or float32: float16 keeps either's weights to its precision or better) and
        computed under float16 autocast with fp16 (V100s), else as the run did."""
        with open(os.path.join(run, "records.json")) as f:
            rec = json.load(f)
        self.run, self.args, self.device, self.node_batch = run, rec["args"], device, node_batch
        self.K = min(depths or len(rec["depths"]), len(rec["depths"]))
        self.steps = [r["steps"] for r in rec["depths"][:self.K]]
        self.records = rec
        self.mean = torch.tensor(rec["mean"], device=device)
        self.window, self.bits = self.args["history_window"], self.args["table_bits"]
        self.cdtype = getattr(torch, self.args.get("centroid_dtype", "float32"))
        total = torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 0
        self.budget = gpu_tables * total
        self.starts: list[Tensor] = []
        self.nets: list[list[dict]] = []
        config = None
        for d in range(self.K):
            row = []
            for t in range(self.steps[d]):
                saved = torch.load(os.path.join(run, "models", f"d{d:03d}_s{t:02d}.pt"), map_location="cpu",
                                   weights_only=False)
                if t == 0:
                    self.starts.append(saved["start"].to(device))
                config = saved["config"]
                row.append({k: v.half() if v.is_floating_point() else v for k, v in saved["state_dict"].items()})
            self.nets.append(row)
        assert config is not None, f"{run} has no networks"
        self.net = make_step_net(config).to(device)
        self.net.fp16 = fp16
        self.net.bf16 = not fp16 and self.args.get("bf16", False)
        self.net.eval()

    @torch.no_grad()
    def code(self, X: Tensor, depths: int | None = None, snapshots=(),
             visit=None) -> tuple[Group, dict[int, Tensor]]:
        """Images X [n, D] (on the device) coded through the first depths depths: their group there (its R the
        remainders), and at each depth in snapshots (the number of depths coded so far) the reconstructions X - R
        then, in float16 on the host. visit(d, cands [n, V, D], tokens [n], shares [n, V]), if given, sees each
        depth's candidates (the cells' predicted means at each image's node), cells and the tree's shares."""
        K = self.K if depths is None else min(depths, self.K)
        g = root_group("replay", X, self.mean, self.window)
        shots = {}
        if 0 in snapshots:
            shots[0] = (X - g.R).half().cpu()
        for d in range(K):
            q = self.starts[d].to(self.cdtype).expand(g.nodes, -1, -1)
            c, p = q, None
            for t, state in enumerate(self.nets[d]):
                self.net.load_state_dict(state)
                c, p = vanswers(self.net, g, q, self.node_batch)
                if t + 1 < len(self.nets[d]):
                    q = c
            assert p is not None
            cells = cells_of(g, q)
            if visit is not None:
                visit(d, c[g.node], cells, p[g.node])
            vdescend(g, cells, split_rows(g, cells, c, p), self.budget, self.bits)
            if d + 1 in snapshots:
                shots[d + 1] = (X - g.R).half().cpu()
        return g, shots

    @torch.no_grad()
    def sample(self, n: int, choose, depths: int | None = None) -> tuple[Tensor, Tensor]:
        """n codes chosen a token at a time, and their reconstructions: at each depth d, choose(d, cands [n, V, D]
        (the candidates: the cells' predicted means at each code's node), shares [n, V] (the tree's there)) gives
        each code's token [n] (long) and the code moves to that cell, by its predicted mean (the networks read only
        the codes' past means, never an image). Returns the tokens [n, K] and the reconstructions [n, D] (the tree's
        mean plus the cells' means), on the device."""
        K = self.K if depths is None else min(depths, self.K)
        g = root_group("sample", self.mean.new_zeros(n, len(self.mean)), self.mean, self.window)
        tokens = torch.zeros(n, K, dtype=torch.long, device=self.device)
        for d in range(K):
            q = self.starts[d].to(self.cdtype).expand(g.nodes, -1, -1)
            c, p = q, None
            for t, state in enumerate(self.nets[d]):
                self.net.load_state_dict(state)
                c, p = vanswers(self.net, g, q, self.node_batch)
                if t + 1 < len(self.nets[d]):
                    q = c
            assert p is not None
            cells = choose(d, c[g.node], p[g.node]).long()
            tokens[:, d] = cells
            vdescend(g, cells, split_rows(g, cells, c, p), self.budget, self.bits)
        return tokens, -g.R  # (R started at -mean and lost each cell's mean)


__all__ = ["VTree"]
