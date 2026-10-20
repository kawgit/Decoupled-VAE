"""The decoupled encoder objective, computed exactly over the code tree.

L(phi) = E_x E_{k~p(k)} E_{z~q(z_{1:k}|x)} E_{y~p(y|z,k)} [d(x, y)],
p(y|z,k) proportional to q(z_{1:k}|y) p_data(y).

Per step we draw a pool of D images. p_data in the posterior is estimated with the pool, leaving the
anchor itself out (y = x has ~zero mass under the population). Every image in the pool can be an
anchor at no extra encoder cost; the first B are used.
"""

import math

import torch


def k_weights(k_max, dist="uniform", geom_ratio=0.7, custom=None):
    """p(k) for k = 1..k_max as a tensor [k_max]."""
    if dist == "uniform":
        w = torch.ones(k_max)
    elif dist == "geometric":
        w = torch.tensor([geom_ratio**k for k in range(k_max)])
    elif dist == "custom":
        if custom is None:
            raise ValueError("--k-dist custom needs --k-weights")
        w = torch.tensor([float(v) for v in custom.split(",")])
        assert len(w) == k_max, f"--k-weights needs {k_max} values"
    else:
        raise ValueError(dist)
    return w / w.sum()


def pairwise_sq_dist(a, b):
    """Mean squared pixel distance between rows of a [B, P] and b [D, P] -> [B, D]."""
    return (a.pow(2).sum(1, keepdim=True) - 2 * a @ b.T + b.pow(2).sum(1)).clamp_min(0) / a.shape[1]


def encoder_objective(prefix_logp, images, n_anchors, weights):
    """
    prefix_logp: list over depth t of [D, V**t] log q(z_{1:t} | pool image), from Encoder.prefix_logprobs.
    images: [D, P] float pixels of the pool. Anchors are images[:n_anchors].
    weights: p(k) [k_max].

    Returns (loss, per_k [k_max] expected distance at each k, k0 = expected distance with no code).
    """
    D = images.shape[0]
    B = n_anchors
    dist = pairwise_sq_dist(images[:B], images)
    dist[torch.arange(B), torch.arange(B)] = 0.0
    rows = torch.arange(D, device=images.device)[:, None]
    anchor_rows = torch.arange(B, device=images.device)[:, None]
    per_k = []
    for lq in prefix_logp[1:]:
        # Leave-one-out softmax over refs, per node, without materializing [B, nodes, D].
        # Shift by the max ref for anchors that aren't the max, and by the runner-up for the anchor
        # that is, so neither the numerator nor the denominator can cancel or underflow.
        top = lq.detach().topk(2, dim=0)
        m1, m2 = top.values
        j1 = top.indices[0]
        is_top_ref = rows == j1  # [D, nodes]
        p1 = (lq - m1).exp()
        p2 = (lq - m2).masked_fill(is_top_ref, -math.inf).exp()
        num1, num2 = dist @ p1, dist @ p2  # [B, nodes]
        anchor_is_top = anchor_rows == j1
        den1 = torch.where(anchor_is_top, torch.ones_like(num1), p1.sum(0) - p1[:B])
        den2 = p2.sum(0).expand_as(num2)
        expected = torch.where(anchor_is_top, num2 / den2, num1 / den1)  # E_{y~p(y|z)} d(x, y) per anchor & node
        per_k.append((lq[:B].exp() * expected).sum(1).mean())
    per_k = torch.stack(per_k)
    k0 = dist.sum(1).mean() / (D - 1)
    loss = (weights.to(per_k.device) * per_k).sum()
    return loss, per_k, k0
