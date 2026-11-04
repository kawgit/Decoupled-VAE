"""The autoencoder's codes as dump_autoencoder.py writes them, and what the later stages read off them."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Codes:
    """One split's codes: each image's sides rare [N, K] (bool), the autoencoder's m [N, K] and r [N, K, D] at
    each node on its path (half precision), the digit labels [N], the training mean [D] (a node's predicted
    mean is it plus the deltas so far) and the autoencoder checkpoint that made them."""
    rare: Tensor
    m: Tensor
    r: Tensor
    labels: Tensor
    mean: Tensor
    ckpt: str

    def __len__(self):
        return len(self.rare)

    @property
    def depths(self):
        return self.rare.shape[1]

    def subset(self, idx):
        return Codes(self.rare[idx], self.m[idx], self.r[idx], self.labels[idx], self.mean, self.ckpt)


def load_codes(path) -> Codes:
    c = torch.load(path, weights_only=False)
    mean = c.get("mean")
    if mean is None:  # dumps from before the mean was kept: the autoencoder's own
        mean = torch.load(c["ckpt"], map_location="cpu", weights_only=False)["mean"]
    return Codes(c["rare"], c["m"], c["r"], c["labels"], mean, c["ckpt"])


def deltas(r: Tensor, m: Tensor, rare: Tensor) -> Tensor:
    """Each token's delta [..., K, D]: r on the rare side, -m r on the common one."""
    return torch.where(rare[..., None], r, -m[..., None] * r)


def bits_of(p: Tensor, rare: Tensor) -> Tensor:
    """The bits [...] each token costs when it is rare with probability p [...]."""
    return -torch.where(rare, p, 1 - p).clamp_min(1e-12).log2()


def own_shares(m: Tensor) -> Tensor:
    """The autoencoder's own probability that a token is rare, p = m / (1 + m)."""
    return m / (1 + m)
