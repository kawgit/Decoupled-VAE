"""The decoder's data and scores (train_decoder.py, benchmark.py): images at their codes'
nodes, a code's bits, and -log2 p(x | code)."""

import math

import torch
import torch.nn.functional as F

from dvae.codes import Codes, bits_of, deltas, own_shares
from dvae.deltatree import Decoder
from dvae.prior import BandedPrior, SidePrior, token_bits


class Leaves:
    """One split's images at their codes' nodes after k tokens, as the decoder reads them: delta histories
    [N, k, D] (half precision), sides and ms [N, k], predicted means [N, D] and grey levels [N, D] (uint8), on
    the CPU. The deltas overwrite codes.r in place (a full dump's r is ~8 GB), so price the codes first."""

    def __init__(self, codes: Codes, images, k, chunk=4096):
        for i in range(0, len(codes), chunk):
            codes.r[i:i + chunk] = deltas(codes.r[i:i + chunk].float(), codes.m[i:i + chunk].float(),
                                          codes.rare[i:i + chunk]).half()
        self.deltas = codes.r[:, :k]
        self.sides = codes.rare[:, :k].float() * 2 - 1
        self.ms = codes.m[:, :k].float()
        self.mu = torch.cat([codes.mean + self.deltas[i:i + chunk].float().sum(1)
                             for i in range(0, len(codes), chunk)])
        self.levels = images.flatten(1)

    def __len__(self):
        return len(self.levels)

    def bits(self, decoder, idx, device):
        """-log2 p(x | code) for images idx [b]."""
        logits = decoder(self.deltas[idx].to(device).float(), self.sides[idx].to(device), self.ms[idx].to(device),
                         self.mu[idx].to(device))
        return F.cross_entropy(logits, self.levels[idx].to(device).long(), reduction="none").sum(1) / math.log(2)


@torch.no_grad()
def pixel_bits(decoder, leaves: Leaves, device, batch):
    """-log2 p(x | code) for every image [N], on the CPU."""
    decoder.eval()
    idx = torch.arange(len(leaves))
    return torch.cat([leaves.bits(decoder, idx[i:i + batch], device).cpu() for i in range(0, len(leaves), batch)])


def code_bits(codes: Codes, k, prior: SidePrior | BandedPrior | None, device):
    """-log2 p(code) for every image's first k tokens [N], on the CPU: under the prior, else the encoder's own
    shares. Call it before building the split's Leaves, which overwrite codes.r."""
    if prior is not None:
        return token_bits(prior, codes.rare, codes.m, codes.r, device)[:, :k].sum(1)
    return bits_of(own_shares(codes.m[:, :k].float()), codes.rare[:, :k]).sum(1)


def load_decoder(ckpt, device):
    """A trained decoder from its checkpoint (train_decoder.py's), in eval mode."""
    decoder = Decoder(**ckpt["config"]).to(device)
    decoder.load_state_dict(ckpt["state_dict"])
    return decoder.eval()
