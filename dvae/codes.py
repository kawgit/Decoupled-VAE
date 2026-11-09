"""The encoder's codes as dump_encoder.py writes them, and what the later stages read off them."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Codes:
    """One split's codes: each image's sides rare [N, K] (bool), the encoder's m [N, K] and r [N, K, D] at
    each node on its path (half precision), the digit labels [N], the training mean [D] (a node's predicted
    mean is it plus the deltas so far) and the encoder checkpoint that made them."""
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
    if mean is None:  # dumps from before the mean was kept: the encoder's own
        mean = torch.load(c["ckpt"], map_location="cpu", weights_only=False)["mean"]
    return Codes(c["rare"], c["m"], c["r"], c["labels"], mean, c["ckpt"])


def deltas(r: Tensor, m: Tensor, rare: Tensor) -> Tensor:
    """Each token's delta [..., K, D]: r on the rare side, -m r on the common one."""
    return torch.where(rare[..., None], r, -m[..., None] * r)


def bits_of(p: Tensor, rare: Tensor) -> Tensor:
    """The bits [...] each token costs when it is rare with probability p [...]."""
    return -torch.where(rare, p, 1 - p).clamp_min(1e-12).log2()


def own_shares(m: Tensor) -> Tensor:
    """The encoder's own probability that a token is rare, p = m / (1 + m)."""
    return m / (1 + m)


@dataclass
class TokenCodes:
    """One split's codes from a refining tree (dvae/refine.py, dump_encoder.py): each image's tokens [N, K]
    (long), the shares of every child at each node on its path [N, K, V] and those children's centroids as
    offsets from the node's predicted mean [N, K, V, D] (half precision), the digit labels [N], the training mean
    [D] (a node's predicted mean is it plus the offsets of the children taken so far) and the encoder checkpoint
    that made them."""
    tokens: Tensor
    shares: Tensor
    offsets: Tensor
    labels: Tensor
    mean: Tensor
    ckpt: str

    def __len__(self):
        return len(self.tokens)

    @property
    def depths(self):
        return self.tokens.shape[1]

    @property
    def V(self):
        return self.shares.shape[2]

    def subset(self, idx):
        return TokenCodes(self.tokens[idx], self.shares[idx], self.offsets[idx], self.labels[idx], self.mean,
                          self.ckpt)

    def taken(self, chunk=4096) -> Tensor:
        """Each token's delta [N, K, D] (half precision): the offset of the child it took."""
        return torch.cat([take(self.offsets[i:i + chunk], self.tokens[i:i + chunk])
                          for i in range(0, len(self), chunk)])

    def taken_shares(self) -> Tensor:
        """Each token's share [N, K]."""
        return take(self.shares, self.tokens)


def take(values: Tensor, tokens: Tensor) -> Tensor:
    """values [..., K, V, *rest] at tokens [..., K]: [..., K, *rest]."""
    idx = tokens.reshape(*tokens.shape, 1, *([1] * (values.dim() - tokens.dim() - 1)))
    return values.gather(tokens.dim(), idx.expand(*tokens.shape, 1, *values.shape[tokens.dim() + 1:])) \
        .squeeze(tokens.dim())


def load_token_codes(path) -> TokenCodes:
    c = torch.load(path, weights_only=False)
    return TokenCodes(c["tokens"], c["shares"], c["offsets"], c["labels"], c["mean"], c["ckpt"])


def own_bits(codes: TokenCodes) -> Tensor:
    """The bits [N, K] each token costs under the encoder's own shares."""
    return -codes.taken_shares().float().clamp_min(1e-12).log2()
