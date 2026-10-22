"""Candidate partitions for one code position, pruning, deduplication, and the decompression objective
evaluated exactly per prefix.

Prefix data comes as weighted entries: each image appears with up to S distinct prefixes, sampled without
replacement from its prefix distribution, each with an importance weight. Weighted sums over an image's
entries estimate sums over its whole prefix distribution (exactly, once S covers every prefix).

A candidate for position t+1 (prefix length t) has a sharing level m in 0..t. Entries are grouped by the
first m tokens of their prefix, and each group gets one V-means codebook, fit with the entries' weights to
their residuals (image minus the weighted mean of its full prefix group). An image's label is the nearest
codeword to its residual in its group's codebook. m = t gives every prefix its own codebook; m = 0 gives
one codebook for all prefixes.

Codewords are labelled by rank along one shared direction (the top principal direction of the residuals),
so token v means roughly the same thing in every group ("aligned"); "random" shuffles the labels per group.
Alignment is a pixel-space heuristic. Everything uses flattened pixels; the objective's distance is the
mean squared pixel distance.
"""

import torch

CHUNK = 4096  # entries per vectorized chunk


def gumbel(shape, generator):
    """Standard Gumbel noise (on CPU)."""
    u = torch.rand(shape, generator=generator, dtype=torch.float64).clamp(1e-12, 1 - 1e-12)
    return -torch.log(-torch.log(u))


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


class GroupMeans:
    """Weighted mean image of each full prefix group; unknown groups use the overall mean."""

    def __init__(self, table, means, overall):
        self.table, self.means, self.overall = table, means, overall

    def residual(self, x, ids):
        pos, hit = self.table.find(ids)
        return x - torch.where(hit[:, None], self.means[pos], self.overall)


class Entries:
    """Weighted (image, prefix) pairs at prefix length `depth`: rows of x_table indexed by img."""

    def __init__(self, x_table, img, ids, w, depth, vocab_size):
        self.x_table, self.img, self.ids, self.w, self.depth, self.V = x_table, img, ids, w, depth, vocab_size
        nodes, self.node_inv = torch.unique(ids, return_inverse=True)
        P = x_table.shape[1]
        wsum = torch.zeros(len(nodes), device=w.device).index_add_(0, self.node_inv, w)
        sums = torch.zeros(len(nodes), P, device=w.device)
        for sl in self.chunks():
            sums.index_add_(0, self.node_inv[sl], w[sl, None] * self.x(sl))
        overall = sums.sum(0) / wsum.sum().clamp_min(1e-12)
        self.group_means = GroupMeans(NodeTable(nodes), sums / wsum.clamp_min(1e-12)[:, None], overall)

    def __len__(self):
        return len(self.img)

    def chunks(self):
        for start in range(0, len(self.img), CHUNK):
            yield slice(start, start + CHUNK)

    def x(self, sl):
        return self.x_table[self.img[sl]]

    def resid(self, sl):
        return self.x(sl) - self.group_means.means[self.node_inv[sl]]


def segment_argmax(scores, segments, n_segments):
    """Index of the highest score within each segment -> [n_segments] (computed on CPU)."""
    scores, segments = scores.double().cpu(), segments.cpu()
    best = torch.full((n_segments,), -torch.inf, dtype=torch.float64).scatter_reduce(0, segments, scores, "amax")
    rows = torch.arange(len(scores))
    winners = scores == best[segments]
    return torch.full((n_segments,), len(scores)).scatter_reduce(0, segments[winners], rows[winners], "amin")


def weighted_top_direction(entries, generator, iters=30):
    """Top principal direction of the weighted residuals (power iteration) -> unit [P]."""
    v = torch.randn(entries.x_table.shape[1], generator=generator).to(entries.w.device)
    for _ in range(iters):
        new = torch.zeros_like(v)
        for sl in entries.chunks():
            r = entries.resid(sl)
            new += r.T @ (entries.w[sl] * (r @ v))
        v = new / new.norm().clamp_min(1e-12)
    return v


def grouped_kmeans(entries, groups, n_groups, V, generator, iters=25):
    """Weighted V-means inside every group at once (k-means++ init) -> centroids [n_groups, V, P]."""
    w, device = entries.w, entries.w.device
    P = entries.x_table.shape[1]
    centroids = torch.zeros(n_groups, V, P, device=device)
    d2 = torch.full((len(entries),), torch.inf, device=device)
    log_w = w.clamp_min(1e-30).log()
    for j in range(V):  # k-means++: first centre by weight, then by weight x squared distance
        score = log_w if j == 0 else (w * d2).clamp_min(1e-30).log()
        pick = segment_argmax(score.cpu() + gumbel(len(entries), generator), groups, n_groups).to(device)
        pick = pick.clamp_max(len(entries) - 1)
        centroids[:, j] = entries.resid(pick)
        for sl in entries.chunks():
            d2[sl] = torch.minimum(d2[sl], (entries.resid(sl) - centroids[groups[sl], j]).pow(2).sum(1))
    for _ in range(iters):
        sums = torch.zeros(n_groups * V, P, device=device)
        mass = torch.zeros(n_groups * V, device=device)
        for sl in entries.chunks():
            r = entries.resid(sl)
            nearest = (r[:, None, :] - centroids[groups[sl]]).pow(2).sum(-1).argmin(1)
            key = groups[sl] * V + nearest
            sums.index_add_(0, key, w[sl, None] * r)
            mass.index_add_(0, key, w[sl])
        new = torch.where(mass[:, None] > 0, sums / mass.clamp_min(1e-12)[:, None], centroids.view(-1, P))
        new = new.view(n_groups, V, P)
        if torch.allclose(new, centroids):
            break
        centroids = new
    return centroids


class ShareRule:
    """Labels for any (image, prefix): nearest codeword of the prefix's m-group codebook to its residual."""

    def __init__(self, name, m, depth, V, group_means, groups, codebooks, perm, fallback=None):
        self.name, self.m, self.depth, self.V = name, m, depth, V
        self.group_means, self.groups, self.codebooks, self.perm = group_means, groups, codebooks, perm
        self.fallback = fallback  # used for m-groups that weren't fitted; None for m = 0 (always fitted)

    def labels(self, x_table, img, ids):
        """x_table [*, P] pixels, img [n] rows of it, ids [n] prefix node ids -> labels [n]."""
        out = torch.empty(len(img), dtype=torch.long, device=ids.device)
        for start in range(0, len(img), CHUNK):
            sl = slice(start, start + CHUNK)
            out[sl] = self._labels(x_table[img[sl]], ids[sl])
        return out

    def _labels(self, x, ids):
        r = self.group_means.residual(x, ids)
        pos, hit = self.groups.find(ids // self.V ** (self.depth - self.m))
        nearest = (r[:, None, :] - self.codebooks[pos]).pow(2).sum(-1).argmin(1)
        out = self.perm[pos, nearest]
        if self.fallback is not None and not bool(hit.all()):
            out = torch.where(hit, out, self.fallback._labels(x, ids))
        return out


def fit_candidate(entries, m, u, generator, labels="aligned", fallback=None):
    """One candidate with sharing level m, fit to the weighted entries."""
    V, depth = entries.V, entries.depth
    group_ids = entries.ids // V ** (depth - m)
    groups, inv = torch.unique(group_ids, return_inverse=True)
    codebooks = grouped_kmeans(entries, inv, len(groups), V, generator)
    if labels == "aligned":
        perm = (codebooks @ u).argsort(-1).argsort(-1)  # codeword -> its rank along the shared direction
    else:
        perm = torch.rand(len(groups), V, generator=generator).argsort(-1).to(codebooks.device)
    return ShareRule(f"m={m}", m, depth, V, entries.group_means, NodeTable(groups), codebooks, perm, fallback)


def node_costs(anchor_table, anchor_img, anchor_ids, anchor_p, ref_table, ref_img, ref_ids, ref_p, ref_w,
               anchor_ref, min_weight=1e-5):
    """Exact decompression objective at this position, per anchor.

    For an anchor with prefix z, the posterior over references for prefix (z, t) weights each reference
    entry by ref_w * 1[its prefix is z] * ref_p[t]. The anchor's expected distance is
    sum_t anchor_p[t] * E_{y ~ posterior(z, t)} d(x, y). For squared distance this needs only, per
    (prefix, token), the total weight, weighted pixel sum and weighted squared norm of the references, so
    it is exact over all references at linear cost.

    Anchors: rows anchor_img of anchor_table with prefix ids anchor_ids and next-token distributions
    anchor_p [nA, V]. References: rows ref_img of ref_table, ids ref_ids, distributions ref_p [nR, V],
    entry weights ref_w [nR]. anchor_ref [nA]: the anchor's own row among the references (-1 if none);
    it is left out of its own posterior. If a (prefix, token) has less than min_weight of reference mass,
    fall back to the prefix's posterior, and failing that to all references. Pass centered pixels for
    precision. Returns [nA] mean squared pixel distance.
    """
    V, P, device = anchor_p.shape[1], anchor_table.shape[1], anchor_p.device
    tokens = torch.arange(V, device=device)
    ref_sq = ref_table.pow(2).sum(1)[ref_img]
    wt = ref_p * ref_w[:, None]  # weight of every (reference entry, token)

    keys, key_inv = torch.unique((ref_ids[:, None] * V + tokens).flatten(), return_inverse=True)
    key_inv = key_inv.view(-1, V)
    W = torch.zeros(len(keys), device=device).index_add_(0, key_inv.flatten(), wt.flatten())
    Q = torch.zeros(len(keys), device=device).index_add_(0, key_inv.flatten(), (wt * ref_sq[:, None]).flatten())
    S = torch.zeros(len(keys), P, device=device)
    groups, group_inv = torch.unique(ref_ids, return_inverse=True)
    Wg = torch.zeros(len(groups), device=device).index_add_(0, group_inv, ref_w)
    Qg = torch.zeros(len(groups), device=device).index_add_(0, group_inv, ref_w * ref_sq)
    Sg = torch.zeros(len(groups), P, device=device)
    for start in range(0, len(ref_img), CHUNK):
        sl = slice(start, start + CHUNK)
        xr = ref_table[ref_img[sl]]
        S.index_add_(0, key_inv[sl].flatten(), (wt[sl, :, None] * xr[:, None, :]).flatten(0, 1))
        Sg.index_add_(0, group_inv[sl], ref_w[sl, None] * xr)
    S_all, Q_all, W_all = Sg.sum(0), Qg.sum(), Wg.sum()
    key_table, group_table = NodeTable(keys), NodeTable(groups)

    out = torch.empty(len(anchor_img), device=device)
    for start in range(0, len(anchor_img), CHUNK):
        sl = slice(start, start + CHUNK)
        xa, ids, pa, self_row = anchor_table[anchor_img[sl]], anchor_ids[sl], anchor_p[sl], anchor_ref[sl]
        sqa = xa.pow(2).sum(1)
        has_self = self_row >= 0
        own_w = torch.where(has_self, ref_w[self_row.clamp_min(0)], 0.0)
        own_p = torch.where(has_self[:, None], wt[self_row.clamp_min(0)], 0.0)

        pos, hit = key_table.find(ids[:, None] * V + tokens)
        Wk = torch.where(hit, W[pos], 0.0) - own_p
        Qk = torch.where(hit, Q[pos], 0.0) - own_p * sqa[:, None]
        Sk = torch.where(hit[..., None], S[pos], 0.0) - own_p[..., None] * xa[:, None, :]
        sum_tok = Wk * sqa[:, None] - 2 * (Sk * xa[:, None, :]).sum(-1) + Qk

        gpos, ghit = group_table.find(ids)
        Wp = torch.where(ghit, Wg[gpos], 0.0) - own_w
        Sp = torch.where(ghit[:, None], Sg[gpos], 0.0) - own_w[:, None] * xa
        Qp = torch.where(ghit, Qg[gpos], 0.0) - own_w * sqa
        sum_grp = Wp * sqa - 2 * (Sp * xa).sum(-1) + Qp
        W_rest = W_all - own_w
        sum_all = W_rest * sqa - 2 * ((S_all - own_w[:, None] * xa) * xa).sum(-1) + (Q_all - own_w * sqa)
        F_grp = torch.where(Wp > min_weight, sum_grp / Wp.clamp_min(min_weight), sum_all / W_rest.clamp_min(min_weight))
        F_tok = torch.where(Wk > min_weight, sum_tok / Wk.clamp_min(min_weight), F_grp[:, None])
        out[sl] = (pa * F_tok).sum(-1) / P
    return out


def candidate_proxy(entries, labels, n_images):
    """In-sample objective of hard labels at this position: weighted mean over entries / images."""
    onehot = torch.nn.functional.one_hot(labels, entries.V).float()
    rows = torch.arange(len(entries), device=labels.device)
    cost = node_costs(entries.x_table, entries.img, entries.ids, onehot, entries.x_table, entries.img, entries.ids,
                      onehot, entries.w, rows)
    return float((entries.w * cost).sum() / n_images)


def weighted_node_ari(a, b, ids, w, V):
    """Adjusted Rand index between labelings a and b inside each prefix group, weighted (label-free)."""
    _, inv = torch.unique(ids, return_inverse=True)
    M = int(inv.max()) + 1 if len(inv) else 0
    table = torch.zeros(M * V * V, device=w.device).index_add_(0, (inv * V + a) * V + b, w).view(M, V, V)

    def pairs(t):
        return t * t / 2  # weighted pair mass; matches n(n-1)/2 up to the self-pair term

    n = table.sum((1, 2))
    s_ab, s_a, s_b = pairs(table).sum((1, 2)), pairs(table.sum(2)).sum(1), pairs(table.sum(1)).sum(1)
    expected = s_a * s_b / pairs(n).clamp_min(1e-12)
    top = (s_a + s_b) / 2
    same = ((s_ab - s_a).abs() < 1e-9) & ((s_ab - s_b).abs() < 1e-9)
    ari = torch.where((top - expected).abs() > 1e-9, (s_ab - expected) / (top - expected), same.float())
    return float((ari * n).sum() / n.sum().clamp_min(1e-12))


def costs_by_group(ids, costs):
    """Sum per-entry costs [n, C] within each prefix group -> (group ids, sums [G, C])."""
    groups, inv = torch.unique(ids, return_inverse=True)
    return groups, torch.zeros(len(groups), costs.shape[1], device=costs.device).index_add_(0, inv, costs)
