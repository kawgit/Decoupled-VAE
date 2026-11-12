"""A train_inner_components.py run's kept tree, replayed on any batch of its inputs: what a later stage reads off
the codes (each depth's children offsets r and -m r, and the token taken), computed as needed instead of stored,
since storing r for every training image at every depth of a deep tree would take hundreds of GB.

Every kept step network is loaded once, onto the device (its weights in bfloat16 when the run computed in it, as
its matrix products did anyway: half the memory); each depth's start is the one its step-0 checkpoint holds. A batch
computes in other shapes than the run's passes over all its images, and in low precision that moves the last bits,
enough to send an image near a split to the other side; given the run's own tokens, the replay follows them
instead, so the codes are the run's and only the offsets are recomputed.
"""

import json
import os

import torch
from torch import Tensor

from dvae.components import StepNet, answers, descend, next_hyperplane, root_group
from dvae.data import load_images
from dvae.deltatree import pixels


class TreeReplay:
    def __init__(self, run: str, device: torch.device, depths: int | None = None, node_batch: int = 4096):
        with open(os.path.join(run, "records.json")) as f:
            rec = json.load(f)
        self.run, self.args, self.device, self.node_batch = run, rec["args"], device, node_batch
        self.K = depths or rec["kept"]
        steps = {r["d"]: r["steps"] for r in rec["depths"]}
        self.steps = [steps[d] for d in range(self.K)]
        self.mean = torch.tensor(rec["mean"], device=device)
        self.window, self.bits = self.args.get("history_window", 0), self.args.get("table_bits", 16)
        bf16, fp16 = self.args.get("bf16", False), self.args.get("fp16", False)
        self.nets: list[list[StepNet]] = []
        self.starts: list[Tensor] = []
        for d in range(self.K):
            row = []
            for t in range(self.steps[d]):
                saved = torch.load(os.path.join(run, "models", f"d{d:03d}_s{t:02d}.pt"), map_location=device,
                                   weights_only=False)
                if t == 0:
                    assert "start" in saved, f"{run}'s d{d:03d}_s00.pt has no start: QA it once (qa_inner_components.py)"
                    self.starts.append(saved["start"].to(device))
                net = StepNet(**saved["config"]).to(device)
                net.load_state_dict(saved["state_dict"])
                net.bf16, net.fp16 = bf16, fp16
                if bf16:  # (autocast multiplies in bfloat16 anyway)
                    net.to(torch.bfloat16)
                net.eval()
                row.append(net)
            self.nets.append(row)
        self.D = int(self.mean.numel())

    def inputs(self, split: str, root: str = "data") -> Tensor:
        """The run's inputs for split as it saw them (pixels with its limits and --flip, or its outer latents),
        [N, D] on the CPU."""
        a = self.args
        if a.get("outer"):
            from dvae.outer import load_outer
            _, lat = load_outer(a["outer"], "cpu")
            return lat[split][:a.get(f"limit_{split}")].float().flatten(1)
        X = pixels(load_images(a["dataset"], split, root)[:a.get(f"limit_{split}")])
        if split == "train" and a.get("flip"):
            X = torch.cat([X, X.view(len(X), *self._shape()).flip(-1).flatten(1)])
        return X

    def _shape(self):
        from dvae.data import SHAPES
        return SHAPES[self.args["dataset"]]

    @torch.no_grad()
    def __call__(self, X: Tensor, tokens: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """For inputs X [B, D]: each depth's children offsets [B, K, 2, D] (half precision; the common side's
        -m r, then the rare side's r) and the tokens taken [B, K] (1: the rare side): tokens [B, K] if given (the
        run's own codes, which the replay then follows), else the replay's own."""
        offsets = torch.empty(len(X), self.K, 2, self.D, dtype=torch.half, device=self.device)
        taken = self._replay(X, tokens, offsets)
        return offsets, taken

    @torch.no_grad()
    def to_host(self, X: Tensor, tokens: Tensor | None = None) -> Tensor:
        """As __call__'s offsets, but written depth by depth into host memory (pinned on CUDA), so a big chunk of
        images (what keeps the GPU busy: the replay costs about the same per batch however many images it holds)
        never needs all its offsets on the GPU at once."""
        offsets = torch.empty(len(X), self.K, 2, self.D, dtype=torch.half,
                              pin_memory=self.device.type == "cuda")
        self._replay(X, tokens, offsets)
        return offsets

    def _replay(self, X: Tensor, tokens: Tensor | None, out: Tensor) -> Tensor:
        g = root_group("replay", X.to(self.device).float(), self.mean, self.window)
        taken = []
        follow = tokens.to(self.device).bool() if tokens is not None else None
        budget = float("inf")
        for d in range(self.K):
            hyper = (self.starts[d].expand(g.nodes, -1), g.R.new_zeros(g.nodes))
            r = m = g.R.new_zeros(0)
            for t, net in enumerate(self.nets[d]):
                r, m = answers(net, g, *hyper, self.node_batch)
                if t + 1 < self.steps[d]:
                    hyper = next_hyperplane(r, m)
            rn, mn = r[g.node], m[g.node]
            out[:, d].copy_(torch.stack([-mn[:, None] * rn, rn], 1).half(), non_blocking=True)
            kept = descend(g, *hyper, r, m, budget, self.bits, None if follow is None else follow[:, d])
            taken.append(kept["side"].long())
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)  # (the copies into host memory are done)
        return torch.stack(taken, 1).to(self.device)

    @torch.no_grad()
    def sample(self, n: int, prior=None, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
        """n codes drawn depth by depth (each token's side from prior, a TokenPrior reading the children offsets
        and tokens so far, or with none from the encoder's own share m / (1 + m)), and the images their nodes
        decode to: (tokens [n, K], images [n, image numbers] in [0, 1], through the outer decoder for a run on an
        outer run's latents)."""
        g = root_group("sample", self.mean.expand(n, -1).clone(), self.mean, self.window)
        offsets = torch.zeros(n, self.K, 2, self.D, dtype=torch.half, device=self.device)
        tokens = torch.zeros(n, self.K, dtype=torch.long, device=self.device)
        for d in range(self.K):
            hyper = (self.starts[d].expand(g.nodes, -1), g.R.new_zeros(g.nodes))
            r = m = g.R.new_zeros(0)
            for t, net in enumerate(self.nets[d]):
                r, m = answers(net, g, *hyper, self.node_batch)
                if t + 1 < self.steps[d]:
                    hyper = next_hyperplane(r, m)
            rn, mn = r[g.node], m[g.node]
            offsets[:, d] = torch.stack([-mn[:, None] * rn, rn], 1).half()
            if prior is None:
                rare = mn.clamp_min(0) / (1 + mn.clamp_min(0))
            else:  # (the prior reads this depth's children and the tokens before it)
                rare = prior(offsets[:, :d + 1].float(), tokens[:, :d + 1])[:, d].float().softmax(-1)[:, 1]
            side = torch.rand(n, device=self.device, generator=generator) < rare
            tokens[:, d] = side.long()
            descend(g, *hyper, r, m, float("inf"), self.bits, side)
        return tokens, self.to_images(self.mean - g.R)

    @torch.no_grad()
    def to_images(self, Z: Tensor) -> Tensor:
        """The run's inputs Z [n, D] (pixels, or an outer run's latents) as images [n, image numbers] in [0, 1]."""
        if not self.args.get("outer"):
            return Z.clamp(0, 1)
        from dvae.outer import decode_latents, load_outer
        if not hasattr(self, "_outer"):
            self._outer = load_outer(self.args["outer"], self.device)
        model, lat = self._outer
        bf16 = self.args.get("bf16", False)
        return decode_latents(model, lat, Z, lambda: torch.autocast(self.device.type, dtype=torch.bfloat16 if bf16
                                                                     else torch.float16, enabled=self.device.type == "cuda"))

    def image_shape(self):
        from dvae.data import SHAPES
        return SHAPES[self.args["dataset"]]
