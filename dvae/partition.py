"""Code posteriors, gains and bank-improvement partitions for the blind knight encoder (see README).

A code's posterior weighs every training image by its held-out chance of producing the code, a product of
slice outputs; weights below a cutoff are dropped. A posterior's spread is the mean squared pixel distance
between two different images drawn from it, which needs only its total weight W, |weighted pixel sum|^2
MM, weighted squared norm Q and sum of squared weights W2. A slice's gain at a code is how much splitting
the code's posterior by the slice's token lowers the spread; a partition's implied gain is the gain of a
slice that reproduces its classes exactly.

Pass centered pixels for precision.
"""

import torch

CHUNK_ELEMENTS = 2**25  # floats in the largest temporary per vectorized chunk
BIG_CODE = 256  # codes with at least this many posterior entries score moves by matmul instead of a gather


def spread(W, MM, Q, W2, P):
    """Spread of weightings from their sums (any matching shapes); only meaningful with two or more images."""
    return 2 * (W * Q - MM) / (P * (W * W - W2)).clamp_min(1e-30)


class CodeSet:
    """The distinct prefixes of C sampled codes, depth by depth, with each prefix's exact posterior.

    Codes are numbered globally, depth by depth, so each depth's codes are contiguous. Per code: depth,
    count (samples with that prefix), used (the slices asked along it), and its posterior's total weight W,
    spread D and support size. Posterior entries (code, image, weight) stay sorted by code. Each sample's
    prefix tokens, the slices that drew them and its code id at every depth are kept too.
    """

    def __init__(self, table, x, cutoff, images):
        """table [N, S, V]: every training image's held-out slice outputs; x [N, P]: centered training images;
        images [C]: the sampled training images. Starts with the empty code, whose posterior is uniform."""
        self.table, self.x, self.cutoff, self.images = table, x, cutoff, images
        self.sq = x.pow(2).sum(1)
        N, C, device = len(x), len(images), x.device
        self.tokens = torch.zeros(C, 0, dtype=torch.long, device=device)
        self.slices = self.tokens.clone()
        self.ids = torch.zeros(C, 1, dtype=torch.long, device=device)
        self.depth = torch.zeros(1, dtype=torch.long, device=device)
        self.count = torch.full((1,), float(C), device=device)
        self.used = [torch.zeros(1, 0, dtype=torch.long, device=device)]
        self.e_code = torch.zeros(N, dtype=torch.long, device=device)
        self.e_img = torch.arange(N, device=device)
        self.e_w = torch.ones(N, device=device)
        self.W, self.D, self.support = self._stats(self.e_code, self.e_img, self.e_w, 1)

    @property
    def n_codes(self):
        return len(self.depth)

    @property
    def max_depth(self):
        return self.tokens.shape[1]

    def codes_at(self, u):
        """Global ids of the depth-u codes."""
        return (self.depth == u).nonzero().squeeze(1)

    def _stats(self, local, img, w, n_codes):
        """Total weight, spread and support size of n_codes posteriors from their entries (local code ids)."""
        dev, P = self.x.device, self.x.shape[1]
        W = torch.zeros(n_codes, device=dev).index_add_(0, local, w)
        W2 = torch.zeros(n_codes, device=dev).index_add_(0, local, w * w)
        Q = torch.zeros(n_codes, device=dev).index_add_(0, local, w * self.sq[img])
        n = torch.bincount(local, minlength=n_codes)
        M = torch.zeros(n_codes, P, device=dev)
        rows = max(1, CHUNK_ELEMENTS // P)
        for i in range(0, len(img), rows):
            M.index_add_(0, local[i:i + rows], w[i:i + rows, None] * self.x[img[i:i + rows]])
        D = torch.where(n >= 2, spread(W, M.pow(2).sum(1), Q, W2, P), 0.0)
        return W, D, n

    def _offsets(self):
        counts = torch.bincount(self.e_code, minlength=self.n_codes)
        return torch.cumsum(counts, 0) - counts, counts

    def add_depth(self, slices, tokens):
        """Extend every sample's code by one token [C], drawn by the given slice [C]."""
        u, dev = self.max_depth + 1, tokens.device
        parent = self.ids[:, -1]
        uniq, inv = torch.unique(parent * self.table.shape[2] + tokens, return_inverse=True)
        n_new, start = len(uniq), self.n_codes
        rep = torch.zeros(n_new, dtype=torch.long, device=dev).scatter_(0, inv, torch.arange(len(inv), device=dev))
        p_code, h, tok = parent[rep], slices[rep], tokens[rep]
        self.used.append(torch.cat([self.used[u - 1][p_code - self.codes_at(u - 1)[0]], h[:, None]], 1))

        # each child's entries: its parent's entries times the slice's probability of the child's token
        offsets, counts = self._offsets()
        per_child = counts[p_code]
        child = torch.repeat_interleave(torch.arange(n_new, device=dev), per_child)
        first = torch.repeat_interleave(torch.cumsum(per_child, 0) - per_child, per_child)
        rows = offsets[p_code][child] + torch.arange(len(child), device=dev) - first
        img = self.e_img[rows]
        w = self.e_w[rows] * self.table[img, h[child], tok[child]]
        keep = w >= self.cutoff
        child, img, w = child[keep], img[keep], w[keep]
        W, D, n = self._stats(child, img, w, n_new)

        self.e_code = torch.cat([self.e_code, child + start])
        self.e_img, self.e_w = torch.cat([self.e_img, img]), torch.cat([self.e_w, w])
        self.W, self.D, self.support = torch.cat([self.W, W]), torch.cat([self.D, D]), torch.cat([self.support, n])
        self.depth = torch.cat([self.depth, torch.full((n_new,), u, device=dev)])
        self.count = torch.cat([self.count, torch.bincount(inv, minlength=n_new).float()])
        self.tokens = torch.cat([self.tokens, tokens[:, None]], 1)
        self.slices = torch.cat([self.slices, slices[:, None]], 1)
        self.ids = torch.cat([self.ids, (inv + start)[:, None]], 1)

    def used_mask(self, codes, n_slices):
        """[len(codes), n_slices]: True for the slices already asked along each code."""
        mask = torch.zeros(len(codes), n_slices, dtype=torch.bool, device=codes.device)
        for u in self.depth[codes].unique().tolist():
            rows = (self.depth[codes] == u).nonzero().squeeze(1)
            used = self.used[u][codes[rows] - self.codes_at(u)[0]]
            if used.shape[1]:
                mask[rows[:, None], used] = True
        return mask

    def slice_gains(self, codes, slices, progress=None):
        """Realized gain of each of the given slices [h] at each code -> [len(codes), len(slices)].

        Codes with fewer than two images of weight gain nothing. A token cell with fewer than two images
        falls back to the code's own spread: the token adds nothing there.
        """
        P, V = self.x.shape[1], self.table.shape[2]
        out = torch.zeros(len(codes), len(slices), device=self.x.device)
        if len(slices) == 0:
            return out
        offsets, counts = self._offsets()
        support, offsets, counts = self.support.tolist(), offsets.tolist(), counts.tolist()
        rows = max(1, CHUNK_ELEMENTS // (len(slices) * V))
        for k, z in enumerate(codes.tolist()):
            if progress is not None:
                progress(k, len(codes))
            if support[z] < 2:
                continue
            start, stop = offsets[z], offsets[z] + counts[z]
            W = torch.zeros(len(slices), V, device=self.x.device)
            W2, Q, n = W.clone(), W.clone(), W.clone()
            M = torch.zeros(len(slices), V, P, device=self.x.device)
            for i in range(start, stop, rows):
                img, w = self.e_img[i:min(i + rows, stop)], self.e_w[i:min(i + rows, stop)]
                A = w[:, None, None] * self.table[img][:, slices]
                A = torch.where(A >= self.cutoff, A, 0.0)
                W += A.sum(0)
                W2 += A.pow(2).sum(0)
                Q += torch.einsum("nhc,n->hc", A, self.sq[img])
                n += (A > 0).sum(0)
                M += torch.einsum("nhc,np->hcp", A, self.x[img])
            cells = torch.where(n >= 2, spread(W, M.pow(2).sum(-1), Q, W2, P), self.D[z])
            out[k] = self.D[z] - (W / self.W[z] * cells).sum(1)
        return out


def kmeanspp(x, V, generator):
    """k-means++ start: V seed images, each drawn in proportion to its squared distance to the nearest seed
    so far, and every image labeled by its nearest seed -> [N]."""
    N = len(x)
    seeds = [int(torch.randint(N, (1,), generator=generator))]
    dist = (x - x[seeds[0]]).pow(2).sum(1)
    for _ in range(1, V):
        seeds.append(int(torch.multinomial(dist.clamp_min(0).cpu() + 1e-12, 1, generator=generator)))
        dist = torch.minimum(dist, (x - x[seeds[-1]]).pow(2).sum(1))
    sx = x[seeds]
    return (x.pow(2).sum(1, keepdim=True) - 2 * x @ sx.T + sx.pow(2).sum(1)).argmin(1)


class PartitionSearch:
    """Hartigan's method on the bank's improvement J over a CodeSet (see README).

    J(labels) = sum_z a_z [G_z(labels) - bar_z]_+, with G_z the partition's implied gain at code z. Only codes
    with two or more images of weight take part; the others can't gain.
    """

    def __init__(self, codes: CodeSet, a, V):
        self.V, self.x, self.sq = V, codes.x, codes.sq
        self.P = codes.x.shape[1]
        active = codes.support >= 2
        self.code_ids = active.nonzero().squeeze(1)
        remap = torch.full((codes.n_codes,), -1, dtype=torch.long, device=a.device)
        remap[self.code_ids] = torch.arange(len(self.code_ids), device=a.device)
        keep = active[codes.e_code]
        self.e_code, self.e_img, self.e_w = remap[codes.e_code[keep]], codes.e_img[keep], codes.e_w[keep]
        self.Wz, self.Dz, self.a = codes.W[self.code_ids], codes.D[self.code_ids], a[self.code_ids]
        self.Z = len(self.code_ids)
        # entries stay sorted by code: big codes get one matmul each when scoring moves, small ones a gather
        counts = torch.bincount(self.e_code, minlength=self.Z)
        starts = torch.cumsum(counts, 0) - counts
        big = counts >= BIG_CODE
        self.big_blocks = list(zip(big.nonzero().squeeze(1).tolist(), starts[big].tolist(), (starts + counts)[big].tolist()))
        self.small_rows = (~big[self.e_code]).nonzero().squeeze(1)

    def stats(self, labels):
        """Per (code, class): W, |M|^2, Q, W2, support, and M itself [Z, V, P]."""
        Z, V, P, dev = self.Z, self.V, self.P, self.x.device
        key = self.e_code * V + labels[self.e_img]
        w = self.e_w
        W = torch.zeros(Z * V, device=dev).index_add_(0, key, w).view(Z, V)
        W2 = torch.zeros(Z * V, device=dev).index_add_(0, key, w * w).view(Z, V)
        Q = torch.zeros(Z * V, device=dev).index_add_(0, key, w * self.sq[self.e_img]).view(Z, V)
        n = torch.bincount(key, minlength=Z * V).view(Z, V)
        M = torch.zeros(Z * V, P, device=dev)
        rows = max(1, CHUNK_ELEMENTS // P)
        for i in range(0, len(key), rows):
            M.index_add_(0, key[i:i + rows], w[i:i + rows, None] * self.x[self.e_img[i:i + rows]])
        M = M.view(Z, V, P)
        return W, M.pow(2).sum(-1), Q, W2, n, M

    def cell_spread(self, W, MM, Q, W2, n, Dz):
        return torch.where(n >= 2, spread(W, MM, Q, W2, self.P), Dz)

    def gains(self, st):
        W, MM, Q, W2, n, _ = st
        return self.Dz - (W / self.Wz[:, None] * self.cell_spread(W, MM, Q, W2, n, self.Dz[:, None])).sum(1)

    def objective(self, G, bar):
        return float((self.a * (G - bar).clamp_min(0)).sum())

    def dots(self, M):
        """Each entry's image . every class sum in its code -> [E, V]."""
        out = torch.empty(len(self.e_code), self.V, device=self.x.device)
        for z, start, stop in self.big_blocks:
            out[start:stop] = self.x[self.e_img[start:stop]] @ M[z].T
        rows = max(1, CHUNK_ELEMENTS // (self.V * self.P))
        for i in range(0, len(self.small_rows), rows):
            r = self.small_rows[i:i + rows]
            out[r] = torch.einsum("ep,evp->ev", self.x[self.e_img[r]], M[self.e_code[r]])
        return out

    def deltas(self, labels, st, G, bar):
        """Exact change in J from moving each image to each class -> [N, V] (0 for its own class)."""
        W, MM, Q, W2, n, M = st
        V, N = self.V, len(self.x)
        out = torch.zeros(N, V, device=self.x.device)
        all_dots = self.dots(M)
        rows = max(1, CHUNK_ELEMENTS // (8 * V))
        for i in range(0, len(self.e_code), rows):
            z, j, w = self.e_code[i:i + rows], self.e_img[i:i + rows], self.e_w[i:i + rows]
            sq, dots = self.sq[j], all_dots[i:i + rows]
            a = labels[j][:, None]
            Dz = self.Dz[z][:, None]
            old = W[z] * self.cell_spread(W[z], MM[z], Q[z], W2[z], n[z], Dz)  # [e, V]
            # its own class, without it
            Wa = W[z].gather(1, a) - w[:, None]
            Da = self.cell_spread(Wa, MM[z].gather(1, a) - 2 * w[:, None] * dots.gather(1, a) + (w * w * sq)[:, None],
                                  Q[z].gather(1, a) - (w * sq)[:, None], W2[z].gather(1, a) - (w * w)[:, None],
                                  n[z].gather(1, a) - 1, Dz)
            # every class, with it
            Wb = W[z] + w[:, None]
            Db = self.cell_spread(Wb, MM[z] + 2 * w[:, None] * dots + (w * w * sq)[:, None], Q[z] + (w * sq)[:, None],
                                  W2[z] + (w * w)[:, None], n[z] + 1, Dz)
            change = (old.gather(1, a) + old - Wa * Da - Wb * Db) / self.Wz[z][:, None]
            g, b = G[z][:, None], bar[z][:, None]
            gain = self.a[z][:, None] * ((g + change - b).clamp_min(0) - (g - b).clamp_min(0))
            gain = gain.scatter(1, a, 0.0)
            out.index_add_(0, j, gain)
        return out

    def run(self, labels, bar, max_rounds, tol=0.0, window=5, progress=None):
        """Hartigan from the given labels [N] against the bar [Z] -> (labels, J, rounds). Stops when no move
        helps, after max_rounds, or when the last `window` rounds together raised J by less than
        window * tol * J. progress(round, J) is called after every round."""
        st = self.stats(labels)
        J = self.objective(self.gains(st), bar)
        history, rounds = [J], 0
        for rounds in range(1, max_rounds + 1):
            delta = self.deltas(labels, st, self.gains(st), bar)
            delta.scatter_(1, labels[:, None], -torch.inf)
            best, target = delta.max(1)
            improving = (best > 1e-9 * max(abs(J), 1e-12)).nonzero().squeeze(1)
            if len(improving) == 0:
                break
            order = improving[best[improving].argsort(descending=True)]
            k = len(order)
            while k >= 1:  # apply the k best moves together; halve until J really rises
                trial = labels.clone()
                trial[order[:k]] = target[order[:k]]
                trial_st = self.stats(trial)
                trial_J = self.objective(self.gains(trial_st), bar)
                if trial_J > J:
                    labels, st, J = trial, trial_st, trial_J
                    break
                k //= 2
            if k == 0:
                break
            if progress is not None:
                progress(rounds, J)
            history.append(J)
            if len(history) > window and J - history[-window - 1] < window * tol * abs(J):
                break
        return labels, J, rounds
