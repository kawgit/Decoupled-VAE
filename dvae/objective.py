"""The decoupled encoder objective, in one form.

L(phi) = E_x sum_k p(k) E_{z~q(z_1:k|x)} F_k(x, z),   F_k(x, z) = E_{y~p(y|z_1:k)} d(x, y),
p(y | z_1:k) proportional to q(z_1:k | y) p_data(y).

Two proposal distributions, each corrected with importance weights:
  q'          what codes are drawn from: q at temperature T (q' = q when T = 1).
  posterior'  what the extra reference images are drawn from (see dvae/references.py); the posterior
              is self-normalized over the pool with weights p_data(y) / s(y), passed as ref_log_w,
              where s mixes posterior' with uniform draws and the anchors.

Each step has B anchors x_i and a pool of D reference images that contains the anchors (each anchor
is left out of its own posterior).

For each anchor and prefix length k the loss is sum_n omega_n F_n over a set of codes n:
  enumerated  every length-k prefix, omega_n = q(prefix | x)                         (exact given the pool)
  sampled     S codes drawn from q', omega_n = (q / q')(prefix) / S                     (unbiased)
combine() turns either into a surrogate whose value is the loss estimate and whose gradient is
  own-choice (REINFORCE) term   stopgrad(omega (F - b)) * grad log q(prefix | x)
  pathwise term                  stopgrad(omega) * grad F
with estimator "full" keeping both, "reinforce" only the first and "pathwise" only the second.
"""

import math

import torch

ESTIMATORS = ("full", "reinforce", "pathwise")


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


def anchor_distances(pool, self_pos):
    """[B, D] distances from each anchor (pool row self_pos) to the pool, exactly 0 at the anchor itself."""
    dist = pairwise_sq_dist(pool[self_pos], pool)
    dist[torch.arange(len(self_pos), device=pool.device), self_pos] = 0.0
    return dist


def tree_posterior(prefix_logp, ref_log_w, dist, self_pos):
    """Enumerated codes: F and the effective number of references for every prefix.

    prefix_logp: list over depth t of [D, V**t] log q(prefix | pool image), from Encoder.prefix_logprobs.
    ref_log_w [D]; dist [B, D] from anchor_distances; self_pos [B]: each anchor's row in the pool.
    Returns one (F [B, V**k], ess [B, V**k]) per k = 1..k_max.

    The leave-one-out softmax over references is done per node without materializing [B, nodes, D]:
    shift by the top reference for anchors that aren't it, and by the runner-up for the anchor that is,
    so neither numerator nor denominator can cancel or underflow.
    """
    D = dist.shape[1]
    rows = torch.arange(D, device=dist.device)[:, None]
    out = []
    for lq in prefix_logp[1:]:
        a = lq + ref_log_w[:, None]
        top = a.detach().topk(2, dim=0)
        m1, m2 = top.values
        j1 = top.indices[0]
        p1 = (a - m1).exp()
        p2 = (a - m2).masked_fill(rows == j1, -math.inf).exp()
        anchor_is_top = self_pos[:, None] == j1
        p1_self = p1[self_pos]
        den1 = torch.where(anchor_is_top, torch.ones_like(p1_self), p1.sum(0) - p1_self)
        den2 = p2.sum(0).expand_as(den1)
        F = torch.where(anchor_is_top, (dist @ p2) / den2, (dist @ p1) / den1)
        with torch.no_grad():
            sq1 = torch.where(anchor_is_top, torch.ones_like(p1_self), (p1**2).sum(0) - p1_self**2)
            sq2 = (p2**2).sum(0).expand_as(den1)
            ess = torch.where(anchor_is_top, den2**2 / sq2, den1**2 / sq1)
        out.append((F, ess))
    return out


def sampled_posterior(logq_refs, ref_log_w, dist, self_pos):
    """Sampled codes: F and the effective number of references.

    logq_refs [B, S, D, K]: per-token log-probs of each anchor's S codes under every pool image.
    Returns F [B, S, K] and ess [B, S, K].
    """
    B, _, D, _ = logq_refs.shape
    self_mask = torch.zeros(B, D, dtype=torch.bool, device=logq_refs.device)
    self_mask[torch.arange(B, device=logq_refs.device), self_pos] = True
    log_post = logq_refs.cumsum(-1) + ref_log_w[None, None, :, None]
    posterior = log_post.masked_fill(self_mask[:, None, :, None], -math.inf).softmax(2)
    F = (posterior * dist[:, None, :, None]).sum(2)
    ess = 1 / posterior.detach().pow(2).sum(2)
    return F, ess


def sampled_terms(logq_self, logq_prime_self, F, baseline=True):
    """Per-k (logq, omega, F, b) for combine(), from S codes per anchor drawn from q'.

    logq_self [B, S, K]: per-token log q of the codes (with grad); logq_prime_self: the same under q'.
    baseline: subtract the mean of the other S-1 samples' w F (leave-one-out), which keeps it unbiased.
    """
    S = logq_self.shape[1]
    logq_cum = logq_self.cumsum(-1)
    w = (logq_cum - logq_prime_self.cumsum(-1)).detach().exp()  # prefix importance weights (q / q')(z_1:k)
    b = None
    if baseline and S > 1:
        v = (w * F).detach()
        b = (v.sum(1, keepdim=True) - v) / (S - 1)
    return [(logq_cum[..., k], w[..., k] / S, F[..., k], None if b is None else b[..., k])
            for k in range(F.shape[-1])]


def tree_terms(anchor_prefix_logp, posteriors):
    """Per-k (logq, omega, F, b) for combine(), from the enumerated tree.

    anchor_prefix_logp: list over depth t of [B, V**t] log q(prefix | anchor) (with grad).
    """
    return [(lq, lq.exp(), F, None) for lq, (F, _) in zip(anchor_prefix_logp[1:], posteriors)]


def combine(terms, weights, estimator="full"):
    """Surrogate loss from per-k terms (logq [B, N], omega [B, N], F [B, N], b [B, N] or None).

    The surrogate's value is sum_k p(k) mean_i sum_n omega F; its gradient keeps the own-choice and/or
    pathwise terms according to estimator. Returns (surrogate, per_k values [K]).
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}")
    surrogate, values = torch.zeros((), device=weights.device), []
    for w_k, (logq, omega, F, b) in zip(weights, terms):
        omega = omega.detach()
        value = (omega * F).sum(1).mean()
        s = value.detach()
        if estimator != "pathwise":
            advantage = (F if b is None else F - b).detach()
            own = (omega * advantage * logq).sum(1).mean()
            s = s + own - own.detach()
        if estimator != "reinforce":
            s = s + value - value.detach()
        surrogate = surrogate + w_k * s
        values.append(value.detach())
    return surrogate, torch.stack(values)


def weighted_ess(terms, ess_per_k):
    """Per-k effective number of references, averaged over anchors with the codes' weights omega."""
    out = []
    for (_, omega, _, _), ess in zip(terms, ess_per_k):
        omega = omega.detach()
        out.append(((omega * ess).sum(1) / omega.sum(1).clamp_min(1e-30)).mean())
    return torch.stack(out)
