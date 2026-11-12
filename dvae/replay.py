"""A train_inner_components.py run's kept tree, replayed on any batch of its inputs: what a later stage reads off
the codes (each depth's children offsets r and -m r, and the token taken), computed as needed instead of stored,
since storing r for every training image at every depth of a deep tree would take hundreds of GB.

Every kept step network is loaded once, onto the device (its weights in bfloat16 when the run computed in it, as
its matrix products did anyway: half the memory); each depth's start is the one its step-0 checkpoint holds. A batch
computes in other shapes than the run's passes over all its images, and in low precision that moves the last bits,
enough to send an image near a split to the other side; given the run's own tokens, the replay follows them
instead, so the codes are the run's and only the offsets are recomputed.
"""

import contextlib
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

    def into_store(self, X: Tensor, tokens: Tensor, store: "CompactStore", rows: Tensor):
        """Replay X [B, D] along its tokens [B, K] (the run's own codes: at each depth every image goes to the side
        its token says), writing each depth's r and m into store's rows [B] (host memory, CompactStore.put),
        from which CompactStore.batches rebuilds them."""
        for _ in self.into_store_steps(X, tokens, store, rows):
            pass

    @torch.no_grad()
    def into_store_steps(self, X: Tensor, tokens: Tensor, store: "CompactStore", rows: Tensor,
                         stream: torch.cuda.Stream | None = None):
        """into_store a depth at a time: a generator that yields after each depth (K + 1 times), so a caller can
        interleave it with other work. Its work runs on stream, if given (made current only while it runs, so
        the caller's own work between steps stays on its own stream)."""
        on = lambda: torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()  # noqa: E731
        with on():
            g = root_group("replay", X.to(self.device).float(), self.mean, self.window)
            follow = tokens.to(self.device).bool()
        yield
        for d in range(self.K):
            with on():
                hyper = (self.starts[d].expand(g.nodes, -1), g.R.new_zeros(g.nodes))
                r = m = g.R.new_zeros(0)
                for t, net in enumerate(self.nets[d]):
                    r, m = answers(net, g, *hyper, self.node_batch)
                    if t + 1 < self.steps[d]:
                        hyper = next_hyperplane(r, m)
                rn, mn = r[g.node], m[g.node]
                store.put(rows, d, rn, mn)
                descend(g, *hyper, r, m, float("inf"), self.bits, follow[:, d])
            yield

    @torch.no_grad()
    def sample(self, n: int, prior=None, generator: torch.Generator | None = None,
               bf16: bool = False) -> tuple[Tensor, Tensor]:
        """n codes drawn depth by depth (each token's side from prior, a TokenPrior reading the splits and tokens
        so far, or with none from the encoder's own share m / (1 + m)), and the images their nodes
        decode to: (tokens [n, K], images [n, image numbers] in [0, 1], through the outer decoder for a run on an
        outer run's latents). bf16: the prior computes in bfloat16."""
        g = root_group("sample", self.mean.expand(n, -1).clone(), self.mean, self.window)
        rs = torch.zeros(n, self.K, self.D, dtype=torch.bfloat16 if bf16 else torch.float32, device=self.device)
        ms = torch.zeros(n, self.K, device=self.device)
        tokens = torch.zeros(n, self.K, dtype=torch.long, device=self.device)
        for d in range(self.K):
            hyper = (self.starts[d].expand(g.nodes, -1), g.R.new_zeros(g.nodes))
            r = m = g.R.new_zeros(0)
            for t, net in enumerate(self.nets[d]):
                r, m = answers(net, g, *hyper, self.node_batch)
                if t + 1 < self.steps[d]:
                    hyper = next_hyperplane(r, m)
            rn, mn = r[g.node], m[g.node]
            rs[:, d], ms[:, d] = rn, mn
            if prior is None:
                rare = mn.clamp_min(0) / (1 + mn.clamp_min(0))
            else:  # (the prior reads this depth's split and the tokens before it)
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=bf16):
                    rare = prior(rs[:, :d + 1], ms[:, :d + 1], tokens[:, :d + 1])[:, d].float().softmax(-1)[:, 1]
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


class CompactStore:
    """Children offsets of n images at K depths, kept in host memory as each depth's r in 4 bits (-7..7 times a
    scale per row, two to a byte) and m: K * (D / 2 + 4) bytes per image, plus the tokens (the offsets in half
    precision would be 4 * K * D). Each row's scale is whichever of its largest entry and a few multiples of its
    rms rounds it closest. The prior reads r through a linear map of its direction and its length: 4 bits keep the
    direction to a cosine of about 0.99 (0.95 for very heavy tails), and a buffer twice the size of 8 bits'. Filled by
    TreeReplay.into_store; batches(...) rebuilds r and m on a device."""

    LEVELS = 7  # (r / scale in -7..7, stored plus 8: 1..15)

    def __init__(self, n: int, K: int, D: int):
        assert D % 2 == 0, D
        self.r = torch.empty(n, K, D // 2, dtype=torch.uint8)
        self.scale = torch.empty(n, K, dtype=torch.half)
        self.m = torch.empty(n, K, dtype=torch.half)
        self.tokens = torch.zeros(n, K, dtype=torch.long)

    def put(self, rows: Tensor, d: int, r: Tensor, m: Tensor):
        """Store r [b, D] and m [b] (on any device) as rows' depth d."""
        r = r.float()
        rms = r.pow(2).mean(-1).sqrt()
        scales = torch.stack([r.abs().amax(-1)] + [rms * a for a in (2.0, 2.5, 3.0, 4.0, 5.0)]).clamp_min(1e-12) \
            / self.LEVELS  # (each row's scale: whichever of these rounds it closest, outliers clipped or not)
        steps = lambda c: (r / c[:, None]).round().clamp(-self.LEVELS, self.LEVELS)  # noqa: E731
        error = torch.stack([(steps(c) * c[:, None] - r).pow(2).sum(-1) for c in scales])
        scale = scales.gather(0, error.argmin(0)[None])[0]
        q = (steps(scale) + 8).to(torch.uint8)
        self.r[rows, d] = (q[:, 0::2] | (q[:, 1::2] << 4)).cpu()
        self.scale[rows, d] = scale.half().cpu()
        self.m[rows, d] = m.half().cpu()

    @staticmethod
    def unpack(packed: Tensor, dtype) -> Tensor:
        """r / scale [..., D] in dtype from its packed bytes [..., D / 2]."""
        return torch.stack([packed & 15, packed >> 4], -1).flatten(-2).to(dtype) - 8

    def __len__(self):
        return len(self.r)

    def batches(self, row_batches: list[Tensor], device, dtype=torch.float32, workers: int = 4):
        """For each rows in row_batches: (r [b, K, D] in dtype, m [b, K], tokens [b, K]) on device.
        A thread gathers the next batches' rows into reused pinned buffers while the device works on this one,
        so copying from host memory (~K * D / 2 bytes an image) overlaps the computing; r's rows, nearly all the
        bytes, are gathered by workers threads at once (one copies only so fast)."""
        import queue
        import threading
        from concurrent.futures import ThreadPoolExecutor

        cuda = torch.device(device).type == "cuda"
        most = max((len(r) for r in row_batches), default=0)
        free: queue.Queue = queue.Queue()
        ready: queue.Queue = queue.Queue()
        for _ in range(3 if cuda else 1):  # (each: pinned buffers for a batch, and the event its last copy recorded)
            free.put([torch.empty((most,) + t.shape[1:], dtype=t.dtype, pin_memory=cuda)
                      for t in (self.r, self.scale, self.m, self.tokens)] + [None])

        pool = ThreadPoolExecutor(workers)

        def gather():
            for rows in row_batches:
                staged = free.get()
                if staged[-1] is not None:
                    staged[-1].synchronize()  # (its last copy to the device is done)
                step = -(-len(rows) // workers)
                list(pool.map(lambda a: torch.index_select(self.r, 0, rows[a:a + step], out=staged[0][a:a + step]),
                              range(0, len(rows), step)))
                for src, dst in zip((self.scale, self.m, self.tokens), staged[1:4]):
                    torch.index_select(src, 0, rows, out=dst[:len(rows)])
                ready.put((staged, len(rows)))
            pool.shutdown(wait=False)

        threading.Thread(target=gather, daemon=True).start()
        for _ in row_batches:
            staged, n = ready.get()
            # (on the CPU, a copy: the buffer is refilled at once)
            r8, scale, m, tokens = (t[:n].to(device, non_blocking=True, copy=not cuda) for t in staged[:4])
            if cuda:
                staged[-1] = torch.cuda.Event()
                staged[-1].record()
            free.put(staged)
            yield self.unpack(r8, dtype) * scale.to(dtype)[..., None], m.float(), tokens

    @staticmethod
    def gb(n: int, K: int, D: int) -> float:
        return n * K * (D // 2 + 12) / 2 ** 30
