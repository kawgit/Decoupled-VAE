"""Reference pools for the encoder objective.

A pool is the B anchors followed by D - B extra references drawn from
    prop(y) = alpha * uniform(y) + (1 - alpha) * posterior'(y)       (never an anchor)
where posterior' is the distribution the ranking is turned into, standing in for the true posterior:
    uniform     posterior' = uniform
    two-towers  posterior' = mean over anchors, and over prefix lengths k weighted by p(k), of
                softmax_y(image_vec(y) . prefix_vecs(anchor's code)[k-1])
The towers are trained to score log q(z_1:k | y) up to a constant, so the softmax needs no extra
temperature; mixing over k matches the posterior's breadth at every prefix length.
The uniform share keeps every image reachable, which the importance weights need.

Treating each pool slot as a draw from the mixture s(y) = (B/D) / N + ((D-B)/D) prop(y), the
reference log-weights are log p_data(y) - log s(y) (up to a constant, which the softmax ignores).
"""

import torch

from .data import batches, to_float

RANKINGS = ("uniform", "two-towers")


def build_pool(anchor_idx, n_train, n_refs, generator, posterior_prime=None, alpha=0.25):
    """Returns (pool_idx [D] with the anchors first, ref_log_w [D]); generator is a CPU generator."""
    device = anchor_idx.device
    B, extra = len(anchor_idx), n_refs - len(anchor_idx)
    is_anchor = torch.zeros(n_train, dtype=torch.bool)
    is_anchor[anchor_idx.cpu()] = True
    uniform = (~is_anchor).double()
    uniform /= uniform.sum()
    if posterior_prime is None:
        prop = uniform
    else:
        pp = posterior_prime.cpu().double().masked_fill(is_anchor, 0.0)  # MPS has no float64
        prop = alpha * uniform + (1 - alpha) * pp / pp.sum() if pp.sum() > 0 else uniform
    extras = torch.multinomial(prop, extra, replacement=True, generator=generator) if extra > 0 else \
        torch.zeros(0, dtype=torch.long)
    pool = torch.cat([anchor_idx.cpu(), extras])
    density = B / n_refs / n_train + extra / n_refs * prop[pool]
    return pool.to(device), (-density.log()).float().to(device)


@torch.no_grad()
def tower_bank(towers, images, chunk=4096):
    """image_vec for every training image [N, dim]."""
    return torch.cat([towers.image_vec(to_float(images[idx]))
                      for idx in batches(len(images), chunk, False, images.device, drop_last=False)])


@torch.no_grad()
def tower_posterior_prime(towers, bank, codes, weights):
    """posterior' for two-towers ranking: mixture over anchors and p(k) of softmax over all images -> [N].

    bank [N, dim] from tower_bank; codes [B, K] (one per anchor); weights p(k) [K].
    """
    B, K = codes.shape
    probs = (bank @ towers.prefix_vecs(codes).flatten(0, 1).T).softmax(0)  # [N, B*K]
    return (probs.view(-1, B, K) * weights).sum(-1).mean(1)


def tower_loss(towers, pool_x, codes, target_logq, ref_log_w, self_pos, weights):
    """Fit softmax_j(image_vec(y_j) . prefix_vecs(z_i)[k-1]) to the posterior over the pool, for every k.

    target_logq [B, D, K]: log q(z_i,1:k | y_j) from the encoder (no grad). Both sides use the pool's
    importance weights, so the towers learn log q(z_1:k | y) up to a per-prefix constant. Returns the
    cross-entropy averaged over anchors and weighted by p(k).
    """
    B, D, _ = target_logq.shape
    mask = torch.zeros(B, D, 1, dtype=torch.bool, device=pool_x.device)
    mask[torch.arange(B, device=pool_x.device), self_pos] = True
    w = ref_log_w[None, :, None]
    logits = torch.einsum("bkc,dc->bdk", towers.prefix_vecs(codes), towers.image_vec(pool_x)) + w
    target = (target_logq.detach() + w).masked_fill(mask, float("-inf")).softmax(1)
    log_model = logits.masked_fill(mask, float("-inf")).log_softmax(1)
    cross_entropy = -(target * log_model.masked_fill(mask, 0.0)).sum(1)  # [B, K]
    return (cross_entropy.mean(0) * weights).sum()
