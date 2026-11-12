"""The inner encoder's codes as train_inner_components.py keeps them, and their bits under the encoder's own shares.
The children offsets the prior reads come from replaying the run's networks (dvae/replay.py), not from here."""

import json
import os

import torch
from torch import Tensor


def take(values: Tensor, tokens: Tensor) -> Tensor:
    """values [..., K, V, *rest] at tokens [..., K]: [..., K, *rest]."""
    idx = tokens.reshape(*tokens.shape, 1, *([1] * (values.dim() - tokens.dim() - 1)))
    return values.gather(tokens.dim(), idx.expand(*tokens.shape, 1, *values.shape[tokens.dim() + 1:])) \
        .squeeze(tokens.dim())


def own_bits(codes: "TreeCodes") -> Tensor:
    """The bits [N, K] each token costs under the encoder's own shares."""
    return -take(codes.shares, codes.tokens).float().clamp_min(1e-12).log2()


class TreeCodes:
    """One split's codes as a train_inner_components.py run keeps them (tree/<split>/dDDD.pt in its folder): tokens
    [N, K] (1: the rare side) and the shares the encoder implies [N, K, 2] (common, rare: m / (1 + m) for the rare
    side). K: the run's kept depths (records.json's "kept") unless given."""

    def __init__(self, folder, split, depths=None):
        with open(os.path.join(folder, "records.json")) as f:
            records = json.load(f)
        K = depths or records["kept"]
        self.folder, self.split = folder, split
        # (read whole, sequentially: rows gathered at random from a network file system would crawl)
        files = [torch.load(os.path.join(folder, "tree", split, f"d{j:03d}.pt"), map_location="cpu",
                            weights_only=False) for j in range(K)]
        N = len(files[0]["side"])
        node, m, sides = torch.zeros(N, dtype=torch.long), [], []
        for f in files:
            m.append(f["m"].float()[node])  # (each image's node's m at this depth)
            sides.append(f["side"].long())
            node = f["node"].long()
        self.tokens = torch.stack(sides, 1)
        m = torch.stack(m, 1)
        self.shares = torch.stack([1 / (1 + m), m / (1 + m)], -1)

    def __len__(self):
        return len(self.tokens)

    @property
    def depths(self):
        return self.tokens.shape[1]

    @property
    def V(self):
        return 2
