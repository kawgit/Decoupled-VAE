"""Anchor partitions for one code position, and the exact marginal decompression loss (see README).

A partition's implied marginal decompression loss: for a partition of a set of (image, code) pairs into V
classes, the marginal decompression loss of an encoder that exactly reproduces (image, code + class):
each pair's expected squared distance to the other pairs with the same code and class.

For anchor image a with code z_a, every reference's code is truncated to s+1 tokens, where s is the length
of the prefix it shares with z_a (the whole code when it matches z_a entirely). References are thereby
grouped finely around the anchor's own code and ever more coarsely further from it. Within each group,
Hartigan's method assigns references to V classes to minimise the implied marginal decompression loss:
moves are taken only when they lower it, so it decreases strictly until no single move helps. Moves are
applied in batches for speed, halving a batch until the total loss drops (a single best move always does).

Any image is then labelled by truncating its code against z_a and choosing the class whose total loss rises
least if the image joins it; codes whose truncated group has no references get class 0. Distances are mean
squared pixel distance; pass centered pixels for precision.

References may carry weights, equal within each group (independent draws from that group, each standing
for the group's mass over the number of draws). A weight only scales its group's loss, so it leaves
Hartigan's moves and the labels unchanged; it puts total_loss() on the scale of the whole population.
"""

import torch

CHUNK_ELEMENTS = 2**25  # floats in the largest temporary per vectorized chunk ([rows, V, P])


def chunk_rows(V, P):
    """Rows per chunk so a [rows, V, P] temporary stays around CHUNK_ELEMENTS floats."""
    return max(1024, CHUNK_ELEMENTS // (V * P))


class NodeTable:
    """Sorted ids with a vectorized lookup."""

    def __init__(self, nodes):
        self.nodes = nodes

    def find(self, ids):
        """-> (row in the table, whether the id is present) for each id."""
        if len(self.nodes) == 0:
            return torch.zeros_like(ids), torch.zeros_like(ids, dtype=torch.bool)
        pos = torch.searchsorted(self.nodes, ids).clamp_max(len(self.nodes) - 1)
        return pos, self.nodes[pos] == ids


def truncated_group(codes, anchor_code, V):
    """Group id of each code [n, t] truncated against anchor_code [t]: t*V for a full match, else
    s*V + (the code's token at position s) where s is the shared prefix length."""
    n, t = codes.shape
    if t == 0:
        return torch.zeros(n, dtype=torch.long, device=codes.device)
    shared = (codes == anchor_code).long().cumprod(1).sum(1)
    token = codes.gather(1, shared.clamp_max(t - 1)[:, None]).squeeze(1)
    return torch.where(shared == t, t * V, shared * V + token)


def cluster_cost(n, S, Q, ng, Sg, Qg, P):
    """Implied marginal decompression loss of one (group, class) cluster from its sufficient statistics.

    n members with pixel sum S [.., P] and squared-norm sum Q, in a group with ng members, sum Sg, sum Qg.
    n >= 2: each member's mean distance to the others, summed = 2 n/(n-1) * SSE.
    n == 1: no other member shares code and class, so (as in node_costs) the member's posterior falls back
    to the rest of its group: its mean distance to the other group members.
    """
    nf = n.float()
    sse = Q - S.pow(2).sum(-1) / nf.clamp_min(1)
    many = 2 * nf / (nf - 1).clamp_min(1) * sse
    others = (ng - 1).float()
    single = (others * Q - 2 * (S * (Sg - S)).sum(-1) + (Qg - Q)) / others.clamp_min(1)
    single = torch.where(others > 0, single, 0.0)
    return torch.where(n >= 2, many, torch.where(n == 1, single, 0.0)) / P


class AnchorPartition:
    """A partition built around one anchor code with Hartigan's method (see the module docstring)."""

    def __init__(self, anchor_code, ref_x, ref_codes, V, generator, max_rounds=200, ref_w=None):
        self.anchor_code, self.V, self.P = anchor_code, V, ref_x.shape[1]
        device = ref_x.device
        gid = truncated_group(ref_codes, anchor_code, V)
        groups, ginv = torch.unique(gid, return_inverse=True)
        self.table, G = NodeTable(groups), len(groups)
        sq = ref_x.pow(2).sum(1)
        self.ng = torch.bincount(ginv, minlength=G)
        w = torch.ones(len(ref_x), device=device) if ref_w is None else ref_w
        self.group_w = torch.zeros(G, device=device).index_add_(0, ginv, w) / self.ng  # each group's (equal) weight
        self.Sg = torch.zeros(G, self.P, device=device).index_add_(0, ginv, ref_x)
        self.Qg = torch.zeros(G, device=device).index_add_(0, ginv, sq)
        labels = torch.randint(V, (len(ref_x),), generator=generator).to(device)
        self.rounds = 0
        for self.rounds in range(1, max_rounds + 1):
            n, S, Q = self._stats(ginv, labels, ref_x, sq, G)
            cost = self._costs(n, S, Q)  # [G, V]
            total = cost.sum()
            # exact change in total loss from moving each reference to each other class
            a = labels
            rem = cluster_cost(n[ginv, a] - 1, S[ginv, a] - ref_x, Q[ginv, a] - sq,
                               self.ng[ginv], self.Sg[ginv], self.Qg[ginv], self.P) - cost[ginv, a]
            add = cluster_cost(n[ginv] + 1, S[ginv] + ref_x[:, None, :], Q[ginv] + sq[:, None],
                               self.ng[ginv, None], self.Sg[ginv, None, :], self.Qg[ginv, None], self.P) - cost[ginv]
            delta = (rem[:, None] + add).masked_fill(torch.nn.functional.one_hot(a, V).bool(), torch.inf)
            best, target = delta.min(1)
            improving = (best < -1e-9 * total.abs().clamp_min(1e-12)).nonzero().squeeze(1)
            if len(improving) == 0:
                break
            order = improving[best[improving].argsort()]
            k = len(order)
            while k >= 1:  # apply the k best moves together; halve until the total really drops
                trial = labels.clone()
                trial[order[:k]] = target[order[:k]]
                if self._costs(*self._stats(ginv, trial, ref_x, sq, G)).sum() < total:
                    labels = trial
                    break
                k //= 2
            if k == 0:
                break
        self.n, self.S, self.Q = self._stats(ginv, labels, ref_x, sq, G)
        self.cost = self._costs(self.n, self.S, self.Q)
        self.ref_labels = labels

    def _stats(self, ginv, labels, x, sq, G):
        key = ginv * self.V + labels
        n = torch.bincount(key, minlength=G * self.V).view(G, self.V)
        S = torch.zeros(G * self.V, self.P, device=x.device).index_add_(0, key, x).view(G, self.V, self.P)
        Q = torch.zeros(G * self.V, device=x.device).index_add_(0, key, sq).view(G, self.V)
        return n, S, Q

    def _costs(self, n, S, Q):
        return cluster_cost(n, S, Q, self.ng[:, None], self.Sg[:, None, :], self.Qg[:, None], self.P)

    def total_loss(self):
        return float((self.cost.sum(1) * self.group_w).sum())

    def labels(self, x, codes):
        """Class for each image x [n, P] with code [n, t]: the class whose total loss rises least if the image
        joins it (class 0 when its truncated group has no references)."""
        rows = chunk_rows(self.V, self.P)
        return torch.cat([self._labels(x[i:i + rows], codes[i:i + rows]) for i in range(0, len(x), rows)]) \
            if len(x) > rows else self._labels(x, codes)

    def _labels(self, x, codes):
        pos, hit = self.table.find(truncated_group(codes, self.anchor_code, self.V))
        sq = x.pow(2).sum(1)
        n, S, Q, cost = self.n[pos], self.S[pos], self.Q[pos], self.cost[pos]  # [m, V, ...]
        ng1, Sg1, Qg1 = self.ng[pos] + 1, self.Sg[pos] + x, self.Qg[pos] + sq  # the group with the image added
        own = cluster_cost(n + 1, S + x[:, None, :], Q + sq[:, None], ng1[:, None], Sg1[:, None, :], Qg1[:, None],
                           self.P) - cost
        # lone members of other classes now also measure their distance to the image
        lone = torch.where(n == 1, cluster_cost(n, S, Q, ng1[:, None], Sg1[:, None, :], Qg1[:, None], self.P) - cost,
                           0.0)
        rise = own + lone.sum(1, keepdim=True) - lone
        return torch.where(hit, rise.argmin(1), 0)


def node_costs(anchor_table, anchor_img, anchor_ids, anchor_p, ref_table, ref_img, ref_ids, ref_p, ref_w,
               anchor_ref, min_weight=1e-5):
    """Exact marginal decompression loss at this position: each anchor's term.

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
    rows = chunk_rows(V, P)
    for start in range(0, len(ref_img), rows):
        sl = slice(start, start + rows)
        xr = ref_table[ref_img[sl]]
        S.index_add_(0, key_inv[sl].flatten(), (wt[sl, :, None] * xr[:, None, :]).flatten(0, 1))
        Sg.index_add_(0, group_inv[sl], ref_w[sl, None] * xr)
    S_all, Q_all, W_all = Sg.sum(0), Qg.sum(), Wg.sum()
    key_table, group_table = NodeTable(keys), NodeTable(groups)

    out = torch.empty(len(anchor_img), device=device)
    for start in range(0, len(anchor_img), rows):
        sl = slice(start, start + rows)
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
