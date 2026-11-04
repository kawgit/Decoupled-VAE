"""Sampling codes from the autoencoder's tree, and helpers the qa scripts share."""

import torch

from dvae.deltatree import D, Frontier, child_key, choose, load_tree, root_frontier, tree_steps
from dvae.prior import BandedPrior, SidePrior


@torch.no_grad()
def sample(decoders, mean, n, steps, batch, tol, generator, prior: SidePrior | BandedPrior | None = None,
           temperature=1.0):
    """n codes drawn down the autoencoder's tree (each depth with its Lloyd steps), as their predicted means after
    every token [n, K, D], and the frontier they end at (its histories: deltas, sides and ms). Each token is rare
    with the autoencoder's own probability m / (1 + m), or with a prior, with the prior's (its logit divided by
    temperature)."""
    device = mean.device
    rs, ms, rares = [], [], []  # the codes so far, as the prior reads them
    fr = root_frontier(mean.new_zeros(0, D), mean)
    fr = Frontier(R=fr.R, node=fr.node, bits=fr.bits.expand(n).clone(), deltas=fr.deltas.expand(n, 0, D),
                  sides=fr.sides.expand(n, 0), ms=fr.ms.expand(n, 0))
    out, x = [], mean.expand(n, D)
    for decoder, depth_steps in zip(decoders, steps):
        s = choose(decoder, fr, depth_steps, batch, tol)
        r, m = s.r, s.m
        p = m / (1 + m)
        if prior is not None:
            rs.append(r.half().float())  # as dump_autoencoder.py stores them
            ms.append(m.half().float())
            unknown = torch.zeros(n, dtype=torch.bool, device=device)  # this token's side, which it doesn't read
            logits = prior(torch.stack(rs, 1), torch.stack(ms, 1), torch.stack(rares + [unknown], 1))
            p = torch.sigmoid(logits[:, -1] / temperature)
        rare = torch.rand(n, generator=generator).to(device) < p
        rares.append(rare)
        delta = torch.where(rare[:, None], r, -m[:, None] * r)
        x = x + delta
        out.append(x)
        rc = rare.cpu()
        fr = Frontier(R=fr.R, node=fr.node, bits=child_key(fr.bits, rc, fr.depth),
                      deltas=torch.cat([fr.deltas, delta.half().cpu()[:, None]], 1),
                      sides=torch.cat([fr.sides, (2 * rc.float() - 1)[:, None]], 1),
                      ms=torch.cat([fr.ms, m.cpu()[:, None]], 1),
                      seed=None if s.u is None else torch.where(rare[:, None], s.u[:, 0], s.u[:, 1]))
    return torch.stack(out, 1), fr


def columns(K, n):
    """n token counts from 1 to K, evenly spread and ending with K (all of them when K <= n)."""
    if K <= n:
        return list(range(1, K + 1))
    return [round(1 + (K - 1) * i / (n - 1)) for i in range(n)]


def load_autoencoder(path, device):
    """A train_autoencoder.py checkpoint as (its depths' decoders, training mean, Lloyd steps per depth, node
    batch, tol)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    decoders = load_tree(ckpt, device)
    a = ckpt["args"]
    return decoders, ckpt["mean"].to(device), tree_steps(ckpt, len(decoders)), a["node_batch"], a.get("tol", 0.0)
