"""Candidate partitions for one code position, label alignment, deduplication, and the decompression
objective evaluated exactly per prefix.

A candidate is a rule that labels any (image, prefix) pair with a next token in 0..V-1. Candidates come
from one family, varying how much the split may depend on the prefix and whether groups are split evenly:

  node             V-means inside each prefix group
  node-balanced    each group split into V equal-size bins along its own top principal direction
  shared           one V-word codebook for all groups, applied to the image minus its group's mean
  shared-balanced  one shared direction; each group split into V equal-size bins along it
  node-unaligned   "node" with labels permuted at random per group (tests whether alignment matters)

Labels are aligned across groups by ranking children along one shared direction (the top principal
direction of the within-group residuals), so token v means roughly the same thing in every group.
That is a pixel-space heuristic, not a guarantee: whether it helps is for the trained val score to say.

Groups too small to fit get the matching shared rule. Everything works on flattened pixels; the
objective's distance is the mean squared pixel distance.
"""

import torch

CANDIDATES = ("node", "node-balanced", "shared", "shared-balanced", "node-unaligned")
LABEL_CHUNK = 8192


def pairwise_sq_dist(a, b):
    """Squared Euclidean distance between rows of a [n, P] and b [m, P] -> [n, m]."""
    return (a.pow(2).sum(1, keepdim=True) - 2 * a @ b.T + b.pow(2).sum(1)).clamp_min(0)


def top_direction(x, generator, iters=30):
    """Leading principal direction of the (already centered) rows of x, by power iteration -> unit [P]."""
    v = torch.randn(x.shape[1], generator=generator).to(x.device)
    for _ in range(iters):
        v = x.T @ (x @ v)
        v = v / v.norm().clamp_min(1e-12)
    return v


def kmeans(x, k, generator, iters=25):
    """k-means++ initialization and Lloyd iterations -> centroids [k, P]; x needs at least k rows."""
    c = [x[int(torch.randint(len(x), (1,), generator=generator))]]
    for _ in range(k - 1):
        d = pairwise_sq_dist(x, torch.stack(c)).min(1).values.cpu()
        probs = d / d.sum() if d.sum() > 0 else torch.full_like(d, 1 / len(d))
        c.append(x[int(torch.multinomial(probs, 1, generator=generator))])
    centroids = torch.stack(c)
    for _ in range(iters):
        assign = pairwise_sq_dist(x, centroids).argmin(1)
        counts = torch.bincount(assign, minlength=k)
        sums = torch.zeros_like(centroids).index_add_(0, assign, x)
        new = torch.where(counts[:, None] > 0, sums / counts.clamp_min(1)[:, None], centroids)
        if torch.allclose(new, centroids):
            break
        centroids = new
    return centroids


class NodeTable:
    """Sorted prefix node ids with a vectorized lookup."""

    def __init__(self, nodes):
        self.nodes = nodes

    def find(self, ids):
        """-> (row in the table, whether the id is present) for each id."""
        if len(self.nodes) == 0:
            return torch.zeros_like(ids), torch.zeros_like(ids, dtype=torch.bool)
        pos = torch.searchsorted(self.nodes, ids).clamp_max(len(self.nodes) - 1)
        return pos, self.nodes[pos] == ids


class Rule:
    name = "rule"

    def labels(self, x, ids):
        """x [N, P] flattened pixels, ids [N] prefix node ids -> next-token labels [N]."""
        return torch.cat([self._labels(x[i:i + LABEL_CHUNK], ids[i:i + LABEL_CHUNK])
                          for i in range(0, len(x), LABEL_CHUNK)]) if len(x) else ids.new_zeros(0)

    def _labels(self, x, ids):
        raise NotImplementedError


class GroupMeans:
    """Mean image of each prefix group; unknown groups use the overall mean."""

    def __init__(self, table, means, overall):
        self.table, self.means, self.overall = table, means, overall

    def residual(self, x, ids):
        pos, hit = self.table.find(ids)
        return x - torch.where(hit[:, None], self.means[pos], self.overall)


def _bins(proj, thresholds):
    """Label = number of thresholds the projection exceeds (0 = lowest along the direction)."""
    return (proj[:, None] > thresholds).sum(1)


class SharedKMeans(Rule):
    name = "shared"

    def __init__(self, group_means, codebook):
        self.group_means, self.codebook = group_means, codebook

    def _labels(self, x, ids):
        return pairwise_sq_dist(self.group_means.residual(x, ids), self.codebook).argmin(1)


class SharedBalanced(Rule):
    name = "shared-balanced"

    def __init__(self, group_means, direction, table, thresholds, overall_thresholds):
        self.group_means, self.direction = group_means, direction
        self.table, self.thresholds, self.overall_thresholds = table, thresholds, overall_thresholds

    def _labels(self, x, ids):
        proj = self.group_means.residual(x, ids) @ self.direction
        pos, hit = self.table.find(ids)
        return _bins(proj, torch.where(hit[:, None], self.thresholds[pos], self.overall_thresholds))


class NodeKMeans(Rule):
    def __init__(self, name, table, centroids, perm, fallback):
        self.name, self.table, self.centroids, self.perm, self.fallback = name, table, centroids, perm, fallback

    def _labels(self, x, ids):
        pos, hit = self.table.find(ids)
        out = self.fallback._labels(x, ids)
        if hit.any():
            c = self.centroids[pos[hit]]  # [n, V, P]
            nearest = ((x[hit][:, None, :] - c) ** 2).sum(-1).argmin(1)
            out[hit] = self.perm[pos[hit], nearest]
        return out


class NodeBalanced(Rule):
    name = "node-balanced"

    def __init__(self, table, means, directions, thresholds, fallback):
        self.table, self.means, self.directions, self.thresholds = table, means, directions, thresholds
        self.fallback = fallback

    def _labels(self, x, ids):
        pos, hit = self.table.find(ids)
        out = self.fallback._labels(x, ids)
        if hit.any():
            p = pos[hit]
            proj = ((x[hit] - self.means[p]) * self.directions[p]).sum(1)
            out[hit] = _bins(proj, self.thresholds[p])
        return out


def make_candidates(x, ids, vocab_size, names, generator):
    """Fit the requested candidate rules to images x [N, P] with prefix node ids [N] -> {name: Rule}."""
    V, P, device = vocab_size, x.shape[1], x.device
    nodes, inv, counts = torch.unique(ids, return_inverse=True, return_counts=True)
    table = NodeTable(nodes)
    means = torch.zeros(len(nodes), P, device=device).index_add_(0, inv, x) / counts[:, None]
    group_means = GroupMeans(table, means, x.mean(0))
    resid = x - means[inv]
    u = top_direction(resid, generator)  # the shared direction labels are aligned along
    qs = torch.linspace(0, 1, V + 1, device=device)[1:-1]

    codebook = kmeans(resid, V, generator)
    shared = SharedKMeans(group_means, codebook[(codebook @ u).argsort()])

    proj = resid @ u
    group_th = torch.stack([torch.quantile(proj[inv == m], qs) for m in range(len(nodes))])
    shared_balanced = SharedBalanced(group_means, u, table, group_th, torch.quantile(proj, qs))
    rules = {"shared": shared, "shared-balanced": shared_balanced}

    if any(n.startswith("node") for n in names):
        fit = (counts >= V).nonzero().squeeze(1).tolist()
        fit_table = NodeTable(nodes[fit])
        cents = torch.zeros(len(fit), V, P, device=device)
        perm_aligned = torch.zeros(len(fit), V, dtype=torch.long, device=device)
        perm_random = torch.zeros_like(perm_aligned)
        dirs = torch.zeros(len(fit), P, device=device)
        bal_th = torch.zeros(len(fit), V - 1, device=device)
        for row, m in enumerate(fit):
            members = x[inv == m]
            c = kmeans(members, V, generator)
            rank = ((c - means[m]) @ u).argsort().argsort()  # child's rank along the shared direction
            cents[row], perm_aligned[row] = c, rank
            perm_random[row] = torch.randperm(V, generator=generator).to(device)
            centered = members - means[m]
            d = top_direction(centered, generator)
            dirs[row] = d if d @ u >= 0 else -d
            bal_th[row] = torch.quantile(centered @ dirs[row], qs)
        rules["node"] = NodeKMeans("node", fit_table, cents, perm_aligned, shared)
        rules["node-unaligned"] = NodeKMeans("node-unaligned", fit_table, cents, perm_random, shared)
        rules["node-balanced"] = NodeBalanced(fit_table, means[fit], dirs, bal_th, shared_balanced)
    return {n: rules[n] for n in names}


def node_ari(a, b, ids, vocab_size):
    """Adjusted Rand index between labelings a and b inside each prefix group, size-weighted (label-free)."""
    V = vocab_size
    _, inv = torch.unique(ids, return_inverse=True)
    M = int(inv.max()) + 1 if len(inv) else 0
    table = torch.zeros(M * V * V, device=a.device).index_add_(
        0, (inv * V + a) * V + b, torch.ones(len(a), device=a.device)).view(M, V, V)

    def pairs(t):
        return t * (t - 1) / 2

    n = table.sum((1, 2))
    s_ab, s_a, s_b = pairs(table).sum((1, 2)), pairs(table.sum(2)).sum(1), pairs(table.sum(1)).sum(1)
    expected = s_a * s_b / pairs(n).clamp_min(1)
    top = (s_a + s_b) / 2
    same = (s_ab == s_a) & (s_ab == s_b)
    ari = torch.where((top - expected).abs() > 1e-9, (s_ab - expected) / (top - expected), same.float())
    return float((ari * n).sum() / n.sum())


def dedup(rules, x, ids, vocab_size, threshold=0.95):
    """Drop candidates whose partition (labels ignored) matches an earlier one with ARI > threshold.

    node-unaligned is the same partition as node by construction, so it is kept whenever node is.
    """
    labels = {name: rule.labels(x, ids) for name, rule in rules.items()}
    kept = []
    for name in rules:
        if name == "node-unaligned":
            if "node" in kept:
                kept.append(name)
            continue
        match = next((k for k in kept if k != "node-unaligned" and
                      node_ari(labels[name], labels[k], ids, vocab_size) > threshold), None)
        if match is None:
            kept.append(name)
        else:
            print(f"  dropping candidate {name}: same partition as {match}")
    return {name: rules[name] for name in kept}


def node_costs(anchor_x, anchor_ids, anchor_p, ref_x, ref_ids, ref_p, anchor_ref, chunk=4096, min_weight=1e-5):
    """Exact decompression objective at this position, per anchor.

    For anchor x with prefix z, the posterior over references for prefix (z, t) weights each reference
    by its probability of that prefix: 1[its prefix is z] * ref_p[t]. The anchor's expected distance is
    sum_t anchor_p[t] * E_{y ~ posterior(z, t)} d(x, y). For squared distance this only needs, per
    (prefix, token), the total weight, weighted pixel sum and weighted squared norm of the references,
    so it is exact over all references at linear cost.

    anchor_x [nA, P], anchor_ids [nA], anchor_p [nA, V]; ref_x [nR, P], ref_ids [nR], ref_p [nR, V];
    anchor_ref [nA]: the anchor's row among the references (-1 if none); it's left out of its own posterior.
    If a (prefix, token) has less than min_weight of reference mass, fall back to the prefix's posterior
    (references with prefix z), and failing that to all references.
    Pass centered pixels (subtract the overall mean) for numerical precision. Returns [nA] mean squared
    pixel distance.
    """
    V, P, device = anchor_p.shape[1], anchor_x.shape[1], anchor_x.device
    tokens = torch.arange(V, device=device)
    ref_sq = ref_x.pow(2).sum(1)

    keys, key_inv = torch.unique((ref_ids[:, None] * V + tokens).flatten(), return_inverse=True)
    w = ref_p.flatten()
    W = torch.zeros(len(keys), device=device).index_add_(0, key_inv, w)
    Q = torch.zeros(len(keys), device=device).index_add_(0, key_inv, w * ref_sq.repeat_interleave(V))
    S = torch.zeros(len(keys), P, device=device)
    for start in range(0, len(w), chunk * V):
        rows = torch.arange(start, min(start + chunk * V, len(w)), device=device)
        S.index_add_(0, key_inv[rows], w[rows, None] * ref_x[rows // V])

    groups, group_inv = torch.unique(ref_ids, return_inverse=True)
    Wg = torch.bincount(group_inv, minlength=len(groups)).float()
    Sg = torch.zeros(len(groups), P, device=device).index_add_(0, group_inv, ref_x)
    Qg = torch.zeros(len(groups), device=device).index_add_(0, group_inv, ref_sq)
    S_all, Q_all, W_all = ref_x.sum(0), ref_sq.sum(), float(len(ref_x))
    key_table, group_table = NodeTable(keys), NodeTable(groups)

    out = torch.empty(len(anchor_x), device=device)
    for start in range(0, len(anchor_x), chunk):
        sl = slice(start, start + chunk)
        xa, ids, pa, self_row = anchor_x[sl], anchor_ids[sl], anchor_p[sl], anchor_ref[sl]
        sqa = xa.pow(2).sum(1)
        own = (self_row >= 0).float()
        own_p = torch.where(own[:, None] > 0, ref_p[self_row.clamp_min(0)], torch.zeros_like(pa))

        pos, hit = key_table.find(ids[:, None] * V + tokens)
        Wk = torch.where(hit, W[pos], 0.0) - own_p
        Qk = torch.where(hit, Q[pos], 0.0) - own_p * sqa[:, None]
        Sk = torch.where(hit[..., None], S[pos], 0.0) - own_p[..., None] * xa[:, None, :]
        sum_tok = Wk * sqa[:, None] - 2 * (Sk * xa[:, None, :]).sum(-1) + Qk

        gpos, ghit = group_table.find(ids)
        Wp = torch.where(ghit, Wg[gpos], 0.0) - own
        Sp = torch.where(ghit[:, None], Sg[gpos], 0.0) - own[:, None] * xa
        Qp = torch.where(ghit, Qg[gpos], 0.0) - own * sqa
        sum_grp = Wp * sqa - 2 * (Sp * xa).sum(-1) + Qp
        W_rest = W_all - own
        sum_all = W_rest * sqa - 2 * ((S_all - own[:, None] * xa) * xa).sum(-1) + (Q_all - own * sqa)
        F_grp = torch.where(Wp > 0.5, sum_grp / Wp.clamp_min(1.0), sum_all / W_rest)
        F_tok = torch.where(Wk > min_weight, sum_tok / Wk.clamp_min(min_weight), F_grp[:, None])
        out[sl] = (pa * F_tok).sum(-1) / P
    return out


def costs_by_group(ids, costs):
    """Sum per-anchor costs [nA, C] within each prefix group -> (group ids, sums [G, C])."""
    groups, inv = torch.unique(ids, return_inverse=True)
    return groups, torch.zeros(len(groups), costs.shape[1], device=costs.device).index_add_(0, inv, costs)
