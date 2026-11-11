"""The decoder's data and scores (train_decoder.py): images at their codes' nodes, a code's bits, and
-log2 p(x | code)."""

import math

import torch
import torch.nn.functional as F

from dvae.codes import LazyRows, TreeCodes, own_bits
from dvae.prior import BandedTokenPrior, TokenPrior, code_token_bits


class TokenLeaves:
    """One split's images at their codes' nodes after k tokens (dvae/codes.py's TreeCodes), as TokenDecoder
    reads them: delta histories [N, k, D] (half precision), tokens [N, k], predicted means [N, D] and grey levels
    [N, D] (uint8), on the CPU."""

    def __init__(self, codes: TreeCodes, images, k, chunk=4096):
        self.tokens = codes.tokens[:, :k]
        # (gathered per batch: a whole split's histories may not fit in memory)
        self.deltas = LazyRows(len(codes), lambda idx: codes.deltas(idx, k))
        self.mu = torch.cat([codes.mean + codes.deltas(torch.arange(i, min(i + chunk, len(codes))), k).float().sum(1)
                             for i in range(0, len(codes), chunk)])
        self.levels = images.flatten(1)

    def __len__(self):
        return len(self.levels)

    def bits(self, decoder, idx, device):
        """-log2 p(x | code) for images idx [b]."""
        logits = decoder(self.deltas[idx].to(device).float(), self.tokens[idx].to(device), self.mu[idx].to(device))
        return F.cross_entropy(logits, self.levels[idx].to(device).long(), reduction="none").sum(1) / math.log(2)


def token_code_bits(codes: TreeCodes, k, prior: TokenPrior | BandedTokenPrior | None, device):
    """-log2 p(code) for every image's first k tokens [N], on the CPU: under the prior, else the encoder's
    implied shares (dvae/codes.py's TreeCodes)."""
    if prior is not None:
        return code_token_bits(prior, codes.offsets, codes.tokens, device)[:, :k].sum(1)
    return own_bits(codes)[:, :k].sum(1)


@torch.no_grad()
def pixel_bits(decoder, leaves: TokenLeaves, device, batch):
    """-log2 p(x | code) for every image [N], on the CPU."""
    decoder.eval()
    idx = torch.arange(len(leaves))
    return torch.cat([leaves.bits(decoder, idx[i:i + batch], device).cpu() for i in range(0, len(leaves), batch)])
