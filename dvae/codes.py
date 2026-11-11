"""The encoder's codes as train_encoder_components.py keeps them, and what the later stages read off them."""

import copy

import torch
from torch import Tensor


def take(values: Tensor, tokens: Tensor) -> Tensor:
    """values [..., K, V, *rest] at tokens [..., K]: [..., K, *rest]."""
    idx = tokens.reshape(*tokens.shape, 1, *([1] * (values.dim() - tokens.dim() - 1)))
    return values.gather(tokens.dim(), idx.expand(*tokens.shape, 1, *values.shape[tokens.dim() + 1:])) \
        .squeeze(tokens.dim())


def own_bits(codes: "TreeCodes") -> Tensor:
    """The bits [N, K] each token costs under the encoder's own shares."""
    return -codes.taken_shares().float().clamp_min(1e-12).log2()


class LazyRows:
    """Rows computed on demand: rows[idx] (an index tensor or a slice) is fn(the image indices) [len, ...]."""

    def __init__(self, n, fn):
        self.n, self.fn = n, fn

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            idx = torch.arange(self.n)[idx]
        return self.fn(torch.as_tensor(idx))


class TreeCodes:
    """One split's codes as train_encoder_components.py keeps them (tree/<split>/dDDD.pt in its folder):
    tokens [N, K] (1: the rare side), the shares the encoder implies [N, K, 2] (common, rare), the children's
    offsets rows offsets[idx] [b, K, 2, D] (half precision, gathered on demand from each depth's per-node r, so
    no [N, K, 2, D] table is ever held), the training mean and the folder. K: the run's kept depths
    (records.json's "kept") unless given."""

    def __init__(self, folder, split, depths=None):
        import json
        import os

        with open(os.path.join(folder, "records.json")) as f:
            records = json.load(f)
        K = depths or records["kept"]
        self.folder, self.split, self.ckpt = folder, split, folder
        self.mean = torch.tensor(records["mean"])
        # (read whole, sequentially: rows gathered at random from a network file system would crawl)
        files = [torch.load(os.path.join(folder, "tree", split, f"d{j:03d}.pt"), map_location="cpu",
                            weights_only=False) for j in range(K)]
        assert all(f["r"] is not None for f in files), f"{folder} kept no r for {split} (--no-store-train-r)"
        N = len(files[0]["side"])
        node, nodes, sides = torch.zeros(N, dtype=torch.long), [], []
        for f in files:
            nodes.append(node)
            sides.append(f["side"].long())
            node = f["node"].long()
        self.node, self.tokens = torch.stack(nodes, 1), torch.stack(sides, 1)
        self.r, self.m = [f["r"] for f in files], [f["m"].float() for f in files]
        m = torch.stack([self.m[j][self.node[:, j]] for j in range(K)], 1)
        self.shares = torch.stack([1 / (1 + m), m / (1 + m)], -1)
        self.offsets = LazyRows(len(self.tokens), self._offsets)

    def __len__(self):
        return len(self.tokens)

    @property
    def depths(self):
        return self.tokens.shape[1]

    @property
    def V(self):
        return 2

    def _offsets(self, idx):
        out = []
        for j in range(self.depths):
            at = self.node[idx, j]
            r, m = self.r[j][at].float(), self.m[j][at]
            out.append(torch.stack([-m[:, None] * r, r], 1))
        return torch.stack(out, 1).half()

    def deltas(self, idx, k=None) -> Tensor:
        """Each token's delta [b, k, D] (half precision): r on the rare side, -m r on the common one."""
        out = []
        for j in range(k or self.depths):
            at = self.node[idx, j]
            r, m = self.r[j][at].float(), self.m[j][at]
            out.append(torch.where(self.tokens[idx, j, None].bool(), r, -m[:, None] * r))
        return torch.stack(out, 1).half()

    def subset(self, idx):
        sub = copy.copy(self)
        sub.node, sub.tokens, sub.shares = self.node[idx], self.tokens[idx], self.shares[idx]
        sub.offsets = LazyRows(len(sub.tokens), sub._offsets)
        return sub

    def taken_shares(self) -> Tensor:
        return take(self.shares, self.tokens)
