"""Slice trees (see README): a code is the sides of a sequence of hyperplanes, each chosen at its node by
Lloyd's algorithm running on a half-space model.

Pixels are dequantized to [0, 1] and centered on the training mean. A node is the intersection of the
oriented half-spaces {x : n . x >= o} along its code; a slice (n, o) asked at a node splits it into the
sides {n . x >= o} (token +1) and {n . x < o} (token -1), the second being the oriented slice (-n, -o).

A density is a mixture of Gaussian components N(mean_i, diag_i + U_i U_i^T). Inside a node each component
is cut by every half-space along the code and kept Gaussian by moment matching, which keeps its covariance
in the form Sigma_i - Sigma_i H^T A H Sigma_i, with H the node's normals and A a small matrix per component.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

D = 784
VAR_FLOOR = 1 / (12 * 256**2)  # the dequantization noise's variance: no pixel varies less
LOG_2PI = math.log(2 * math.pi)
ELEMENTS = 2**23  # floats in the largest temporary of a chunked computation


def dequantize(images, generator):
    """uint8 [N, 28, 28] -> float [N, 784]: each grey level spread uniformly over its 1/256 bin."""
    return (images.float() + torch.rand(images.shape, generator=generator)).div(256).flatten(1)


def normal_side(alpha):
    """Phi(alpha) and the inverse Mills ratio phi(alpha) / Phi(alpha), stable for very negative alpha."""
    Phi = 0.5 * torch.erfc(-alpha / math.sqrt(2))
    phi = torch.exp(-0.5 * alpha.square()) / math.sqrt(2 * math.pi)
    a = alpha.clamp_max(-5.0)
    tail = -a / (1 - a.pow(-2) + 3 * a.pow(-4))
    return Phi, torch.where(alpha < -5, tail, phi / Phi.clamp_min(1e-30))


# ===== Gaussians with diagonal-plus-low-rank covariance =====

def lowrank_parts(diag, U):
    """For Sigma = diag(diag) + U U^T ([..., D], [..., D, r]): 1/diag, Z = U / diag, the Cholesky factor L
    of I + U^T Z, and log det Sigma."""
    inv = 1 / diag
    Z = U * inv[..., None]
    M = torch.eye(U.shape[-1], device=U.device) + U.mT @ Z
    L = torch.linalg.cholesky(M)
    logdet = diag.log().sum(-1) + 2 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(-1)
    return inv, Z, L, logdet


def lowrank_quad(v, inv, Z, L):
    """v^T Sigma^-1 v for v [..., D], by the Woodbury identity."""
    t = tri_inverse(L) @ (Z.mT @ v[..., None])
    return (v.square() * inv).sum(-1) - t.square().sum((-2, -1))


def tri_inverse(L):
    """The inverse of small lower-triangular factors [..., r, r] (cheaper than solving against D columns)."""
    eye = torch.eye(L.shape[-1], device=L.device).expand_as(L)
    return torch.linalg.solve_triangular(L, eye, upper=False)


def lowrank_logpdf(x, mean, diag, U):
    inv, Z, L, logdet = lowrank_parts(diag, U)
    return -0.5 * (D * LOG_2PI + logdet + lowrank_quad(x - mean, inv, Z, L))


def gaussian_ce(t_mean, t_diag, t_U, mean, diag, U):
    """E[-log N(x; mean, diag + U U^T)] for x ~ N(t_mean, t_diag + t_U t_U^T), per pixel."""
    inv, Z, L, logdet = lowrank_parts(diag, U)
    W = tri_inverse(L) @ Z.mT                                                    # L^-1 Z^T [..., r, D]
    trace = ((inv - W.square().sum(-2)) * t_diag).sum(-1)                         # tr(Sigma^-1 t_diag)
    trace = trace + (t_U.square() * inv[..., None]).sum((-2, -1)) - (W @ t_U).square().sum((-2, -1))
    quad = lowrank_quad(t_mean - mean, inv, Z, L)
    return 0.5 * (D * LOG_2PI + logdet + trace + quad) / D


def gaussian_entropy(diag, U):
    """The entropy of N(., diag + U U^T), per pixel."""
    return 0.5 * (D * (LOG_2PI + 1) + lowrank_parts(diag, U)[3]) / D


def gaussian_kl(t_mean, t_diag, t_U, mean, diag, U):
    """KL(N(t_mean, t_diag + t_U t_U^T) || N(mean, diag + U U^T)), per pixel: gaussian_ce minus the target's
    entropy, so 0 for a perfect fit whatever the target."""
    return gaussian_ce(t_mean, t_diag, t_U, mean, diag, U) - gaussian_entropy(t_diag, t_U)


def bernoulli_kl(target, logit):
    """KL(Bernoulli(target) || Bernoulli(sigmoid(logit))): the cross-entropy minus the target's entropy."""
    entropy = -(target * target.clamp_min(1e-12).log() + (1 - target) * (1 - target).clamp_min(1e-12).log())
    return F.binary_cross_entropy_with_logits(logit, target, reduction="none") - entropy


# ===== hashed randomness: the same node always gets the same starts =====

_C1, _C2, _C3 = (c - (1 << 64) for c in (0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB))


def _shr(x, s):
    return (x >> s) & ((1 << (64 - s)) - 1)


def splitmix64(x):
    """A 64-bit integer hash (int64 CPU tensors, wrapping arithmetic)."""
    z = x + _C1
    z = (z ^ _shr(z, 30)) * _C2
    z = (z ^ _shr(z, 27)) * _C3
    return z ^ _shr(z, 31)


def node_seeds(bits, depth, salt=0):
    """A seed per node from its code (bits [B] int64, token +1 at depth u sets bit u) and depth."""
    return splitmix64(bits * 64 + depth + salt * (1 << 40))


def hashed_normals(seeds, n):
    """[B, n] standard normals, a fixed function of each seed."""
    j = torch.arange(2 * n, dtype=torch.int64)
    h = splitmix64(seeds[:, None] ^ splitmix64(j)[None])
    u = (_shr(h, 11).double() + 0.5) / 2.0**53
    u1, u2 = u[:, :n], u[:, n:]
    return (torch.sqrt(-2 * torch.log(u1)) * torch.cos(2 * math.pi * u2)).float()


# ===== the components of a density =====

class Bank:
    """Gaussian components N(mean_i, diag(diag_i) + U_i U_i^T). diag is [M, D], or [D] when every component
    shares it; the low-rank parts are stored in half precision."""

    def __init__(self, mean, diag, U):
        self.mean, self.U = mean.float(), U.half()
        self.shared = diag.dim() == 1
        self.diag = diag.float() if self.shared else diag.half()

    @property
    def rank(self):
        return self.U.shape[-1]

    def parts(self, comp):
        """Components comp [B, c] -> (Diag, U [B, c, D, r])."""
        return Diag(self.diag if self.shared else self.diag[comp].float()), self.U[comp].float()

    @property
    def kernel_diag(self):
        """A typical component's diagonal [D]: the density's resolution, below which no target's variance goes."""
        return self.diag if self.shared else self.diag.float().mean(0)

    def diag_of(self, comp):
        return self.diag.expand(*comp.shape, D) if self.shared else self.diag[comp].float()


class Diag:
    """The diagonal parts of a chunk of components: one [D] shared by all, or [B, c, D]. Its methods are
    the reductions the cut formulas need, done without forming [B, c, D, Q] arrays."""

    def __init__(self, d):
        self.d = d

    def quad(self, V):
        """sum_d diag_d V_dq^2 for V [B, D, Q] -> [B, c or 1, Q]."""
        if self.d.dim() == 1:
            return torch.einsum("d,bdq->bq", self.d, V.square())[:, None]
        return torch.einsum("bcd,bdq->bcq", self.d, V.square())

    def H(self, H, V):
        """H diag V for H [B, k, D], V [B, D, Q] -> [B, c or 1, k, Q]."""
        if self.d.dim() == 1:
            return ((H * self.d) @ V)[:, None]
        return torch.einsum("bkd,bcd,bdq->bckq", H, self.d, V)

    def weighted(self, w):
        """sum_c w_cq diag_c for w [B, c, Q] -> [B, Q, D]."""
        if self.d.dim() == 1:
            return w.sum(1)[..., None] * self.d
        return torch.einsum("bcq,bcd->bqd", w, self.d)

    def weighted_H(self, H, G):
        """sum_c diag_c * (H^T G_c) for H [B, k, D], G [B, c, k, Q] -> [B, Q, D]."""
        if self.d.dim() == 1:
            return torch.einsum("bkd,bkq->bqd", H, G.sum(1)) * self.d
        return torch.einsum("bkd,bcd,bckq->bqd", H, self.d, G)

    def own(self):
        return self.d


def nearest(Q, X, k, exclude_self=False):
    """Indices [len(Q), k] of the k nearest rows of X to each row of Q (X itself when exclude_self)."""
    out = []
    sq = X.square().sum(1)
    step = max(1, ELEMENTS // len(X))
    for i in range(0, len(Q), step):
        q = Q[i:i + step]
        dist = sq[None] - 2 * q @ X.T
        if exclude_self:
            dist[torch.arange(len(q)), torch.arange(i, i + len(q))] = math.inf
        out.append(dist.topk(k, largest=False).indices)
    return torch.cat(out)


def local_directions(X, neighbours, r):
    """Each image's top r directions to its neighbours, scaled by their spread: U [N, D, r]."""
    out = []
    k = neighbours.shape[1]
    step = max(1, ELEMENTS // (k * D))
    for i in range(0, len(X), step):
        diff = X[neighbours[i:i + step]] - X[i:i + step, None]                 # [b, k, D]
        vals, vecs = torch.linalg.eigh(diff @ diff.mT)                            # ascending, [b, k]
        vals, vecs = vals[:, -r:].clamp_min(1e-12), vecs[:, :, -r:]
        dirs = (vecs.mT @ diff) / vals.sqrt()[..., None]                          # unit directions [b, r, D]
        out.append((dirs.mT * (vals.sqrt() / math.sqrt(k))[:, None]).half())
    return torch.cat(out)


def mixture_logpdf(Y, bank, candidates, scale=1.0, var=None):
    """log of the bank's equal-weight mixture at each row of Y [n, D], summing over candidates [n, c].
    scale multiplies the low-rank parts, and var [D], when given, replaces every component's diagonal."""
    out = []
    c = candidates.shape[1]
    step = max(1, ELEMENTS // (c * D * bank.rank))
    for i in range(0, len(Y), step):
        idx = candidates[i:i + step]
        diag = bank.diag_of(idx) if var is None else var.expand(*idx.shape, D)
        lp = lowrank_logpdf(Y[i:i + step, None], bank.mean[idx], diag, bank.U[idx].float() * scale)
        out.append(lp.logsumexp(1) - math.log(len(bank.mean)))
    return torch.cat(out)


def bits_per_dim(log_density):
    """Bits per pixel of 8-bit images, from a log density over [0, 1] pixels."""
    return -log_density / (D * math.log(2)) + 8


def fit_parzen(X, Y, k=32, r=8, candidates=64, log=print):
    """Round 0's density: a Gaussian on every training image. Its low-rank part is the local principal
    directions of its k nearest neighbours, and its diagonal one per-pixel variance shared by every image,
    proportional to the pixel's variance over the training set. Both scales are chosen on held-out Y."""
    U = local_directions(X, nearest(X, X, k, exclude_self=True), r)
    pixel = X.var(0)
    probe = Bank(X, torch.zeros(D, device=X.device), U)
    cand = nearest(Y, X, candidates)
    best = (-math.inf, 1.0, 1.0)
    for scale in (0.35, 0.5, 0.7, 1.0, 1.4, 2.0):
        for var in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
            lp = mixture_logpdf(Y, probe, cand, scale, (var * pixel).clamp_min(VAR_FLOOR)).mean().item()
            best = max(best, (lp, scale, var))
    lp, scale, var = best
    log(f"parzen: scale {scale}, variance {var} x the pixel's, held-out {bits_per_dim(torch.tensor(lp)).item():.4f} bits/dim")
    diag = (var * pixel).clamp_min(VAR_FLOOR)
    del probe
    return Bank(X, diag, U.mul_(scale)), scale, var


# ===== nodes: components cut by the half-spaces along a code =====

@dataclass
class Nodes:
    comp: Tensor  # [B, C] component indices into the bank (padding: weight 0)
    w: Tensor     # [B, C] each component's mass inside the node
    m: Tensor     # [B, C, D] its moment-matched mean
    A: Tensor     # [B, C, k, k] its covariance is Sigma - Sigma H^T A H Sigma
    H: Tensor     # [B, k, D] the node's oriented normals
    o: Tensor     # [B, k] and offsets: the node is {x : H x >= o}

    def select(self, idx):
        return Nodes(self.comp[idx], self.w[idx], self.m[idx], self.A[idx], self.H[idx], self.o[idx])

    @staticmethod
    def cat(parts):
        C = max(p.comp.shape[1] for p in parts)

        def pad(t, value=0):
            return F.pad(t, [0, 0] * (t.dim() - 2) + [0, C - t.shape[1]], value=value)
        return Nodes(torch.cat([pad(p.comp) for p in parts]), torch.cat([pad(p.w) for p in parts]),
                     torch.cat([pad(p.m) for p in parts]), torch.cat([pad(p.A) for p in parts]),
                     torch.cat([p.H for p in parts]), torch.cat([p.o for p in parts]))


def root_nodes(bank, device):
    M = len(bank.mean)
    return Nodes(torch.arange(M, device=device)[None], torch.full((1, M), 1 / M, device=device),
                 bank.mean[None], torch.zeros(1, M, 0, 0, device=device),
                 torch.zeros(1, 0, D, device=device), torch.zeros(1, 0, device=device))


def _query_terms(bank, nodes, c0, c1, qn, qo):
    """Per component of nodes[:, c0:c1] and query (qn [B, Q, D], qo [B, Q]), with h = H Sigma_0 n and
    g = A h: Sigma_k n = Sigma_0 (n - H^T g), so U^T Sigma_k n's U part is t = U^T n - (H U)^T g and
    s^2 = n^T Sigma_0 n - h^T g. Returns the parts, Phi, lambda, s, alpha [B, c, Q], t [B, c, r, Q], g
    [B, c, k, Q] (None at the root) and H U [B, c, k, r]."""
    comp = nodes.comp[:, c0:c1]
    dg, U = bank.parts(comp)
    qT = qn.mT
    Uq = torch.einsum("bcdr,bdq->bcrq", U, qT)
    s0 = dg.quad(qT) + Uq.square().sum(2)
    s2 = s0
    g = HU = None
    t = Uq
    if nodes.H.shape[1]:
        HU = torch.einsum("bkd,bcdr->bckr", nodes.H, U)
        h = dg.H(nodes.H, qT) + HU @ Uq
        g = nodes.A[:, c0:c1] @ h
        s2 = s0 - (h * g).sum(2)
        t = Uq - HU.mT @ g
    # After many cuts nearly parallel to the query, s0 - h^T g can cancel to 0 or below in float32; keep
    # at least a millionth of the component's untruncated variance along the query.
    s = torch.maximum(s2, 1e-6 * s0).clamp_min(1e-12).sqrt()
    alpha = (torch.einsum("bcd,bqd->bcq", nodes.m[:, c0:c1], qn) - qo[:, None]) / s
    Phi, lam = normal_side(alpha)
    return dg, U, Phi, lam, s, alpha, t, g, HU


def _comp_step(nodes, Q, r):
    """Components per chunk, so that a chunk's largest arrays hold about ELEMENTS floats."""
    k = nodes.H.shape[1]
    per = D * (r + 2 + k) + Q * (r + 2 * k + 4)
    return max(1, ELEMENTS // (len(nodes.comp) * per))


def side_moments(bank, nodes, qn, qo):
    """The density's mass, first moment, and second moment of the distance u = qn . x - qo, on the side
    {qn . x >= qo} of each query: [B, Q], [B, Q, D], [B, Q]."""
    B, Q = qo.shape
    mass = torch.zeros(B, Q, device=qn.device)
    first = torch.zeros(B, Q, D, device=qn.device)
    second = torch.zeros(B, Q, device=qn.device)
    step = _comp_step(nodes, Q, bank.rank)
    for c0 in range(0, nodes.comp.shape[1], step):
        c1 = c0 + step
        dg, U, Phi, lam, s, alpha, t, g, _ = _query_terms(bank, nodes, c0, c1, qn, qo)
        wPhi = nodes.w[:, c0:c1, None] * Phi
        coef = wPhi * lam / s
        mass += wPhi.sum(1)
        second += (wPhi * s.square() * (1 + alpha * (alpha + lam))).sum(1)  # E[u^2] of a truncated Gaussian
        first += torch.einsum("bcq,bcd->bqd", wPhi, nodes.m[:, c0:c1])
        first += dg.weighted(coef) * qn + torch.einsum("bcdr,bcrq->bqd", U, coef[:, :, None] * t)
        if g is not None:
            first -= dg.weighted_H(nodes.H, coef[:, :, None] * g)
    return mass, first, second


def cut(bank, nodes, qn, qo, cap, protect=None):
    """The nodes' sides {qn . x >= qo} (qn [B, D], qo [B]), keeping at most cap components each: the
    heaviest, after any listed in protect ([B, P] component indices, -1 for none), which are always kept
    while they fit. Round 0 protects the kernels of the real images still in each side, so the density
    never drops the mass around an image that is there."""
    w, m, A = [], [], []
    k = nodes.H.shape[1]
    step = _comp_step(nodes, 1, bank.rank)
    for c0 in range(0, nodes.comp.shape[1], step):
        c1 = c0 + step
        dg, U, Phi, lam, s, alpha, t, g, _ = _query_terms(bank, nodes, c0, c1, qn[:, None], qo[:, None])
        Phi, lam, s, alpha, t = Phi[..., 0], lam[..., 0], s[..., 0], alpha[..., 0], t[..., 0]
        v = qn[:, None].expand(-1, Phi.shape[1], -1)
        if g is not None:
            g = g[..., 0]
            v = v - torch.einsum("bkd,bck->bcd", nodes.H, g)
        Sn = dg.own() * v + (U @ t[..., None])[..., 0]                      # Sigma_k n [B, c, D]
        c = lam * (lam + alpha) / s.square()                                  # the variance lost along Sigma n
        gk = g if g is not None else Phi.new_zeros(*Phi.shape, 0)
        vec = torch.cat([-gk, Phi.new_ones(*Phi.shape, 1)], -1)
        mc = nodes.m[:, c0:c1] + (lam / s)[..., None] * Sn
        Ac = F.pad(nodes.A[:, c0:c1], (0, 1, 0, 1)) + c[..., None, None] * vec[..., :, None] * vec[..., None, :]
        # A component left with under a millionth of its mass is retired: there float32 cancellation can
        # blow up its moment-matched mean, and a weight of 0 times an infinite mean would poison every sum.
        dead = (Phi < 1e-6) | ~torch.isfinite(mc).all(-1) | ~torch.isfinite(Ac).flatten(-2).all(-1)
        w.append(torch.where(dead, 0.0, nodes.w[:, c0:c1] * Phi))
        m.append(torch.where(dead[..., None], 0.0, mc))
        A.append(torch.where(dead[..., None, None], 0.0, Ac))
    assert k == nodes.A.shape[-1]
    w, m, A = torch.cat(w, 1), torch.cat(m, 1), torch.cat(A, 1)
    keep = min(cap, w.shape[1])
    key = w
    if protect is not None and protect.shape[1]:
        M = len(bank.mean)
        rows = torch.arange(len(w), device=w.device)[:, None]
        wanted = (rows * M + protect)[protect >= 0]
        kept = torch.isin(rows * M + nodes.comp, wanted) & (w > 0)
        key = torch.where(kept, w + w.sum(1, keepdim=True) + 1, w)
    order = key.topk(keep, dim=1).indices
    w = w.gather(1, order)
    w = torch.where(w >= 1e-7 * w.sum(1, keepdim=True), w, 0.0)
    gather = order[..., None]
    return Nodes(nodes.comp.gather(1, order), w, m.gather(1, gather.expand(-1, -1, D)),
                 A.gather(1, gather[..., None].expand(-1, -1, *A.shape[2:])),
                 torch.cat([nodes.H, qn[:, None]], 1), torch.cat([nodes.o, qo[:, None]], 1))


def combined_sides(bank, nodes, real, real_mask, alpha, qn, qo):
    """Combined targets on the side of each query (qn [B, Q, D], qo [B, Q]): the n_z real images in a node
    (real [B, R, D], real_mask [B, R]) mixed with alpha [B] pseudo-images from the density inside it.
    Returns the side's share, its mean, the density's share alone, the real images on the side, and the
    mean and variance of the distance u = qn . x - qo on the side."""
    mass, first, second = side_moments(bank, nodes, qn, qo)
    total = nodes.w.sum(1).clamp_min(1e-30)
    density_share = mass / total[:, None]
    u_real = torch.einsum("brd,bqd->brq", real, qn) - qo[:, None]
    side = ((u_real >= 0) & real_mask[..., None]).float()
    n_side = side.sum(1)
    n_z = real_mask.float().sum(1)
    pseudo = alpha[:, None] * density_share
    share = (pseudo + n_side) / (alpha + n_z).clamp_min(1e-30)[:, None]
    mean = (alpha[:, None, None] * first / total[:, None, None] + torch.einsum("brq,brd->bqd", side, real))
    mean = mean / (pseudo + n_side).clamp_min(1e-12)[..., None]
    u2 = (alpha[:, None] * second / total[:, None] + (side * u_real.square()).sum(1)) / (pseudo + n_side).clamp_min(1e-12)
    u_mean = (qn * mean).sum(-1) - qo
    return share, mean, density_share, n_side, u_mean, (u2 - u_mean.square()).clamp_min(VAR_FLOOR)


def side_covariance(bank, nodes, pseudo, real, real_mask, mean, r, generator, iters=3, extra=4, floor=VAR_FLOOR, group=1):
    """The combined covariance of nodes that are already cut to one side: pseudo [B] pseudo-images from the
    density, mixed with the real images there, around the combined mean [B, D]. Compressed to a diagonal
    plus its top r eigenvectors: (diag [B, D], U [B, D, r]). The diagonal is at least floor (a number or
    [D]): a side holding a lone real image and almost none of the density's mass would otherwise collapse
    onto that image. With group > 1, the nodes come in runs of group sides of one node each, real [B / group,
    R, D] holds each node's real images once and real_mask [B, R] marks the ones on each side."""
    B = len(mean)
    weight = nodes.w / nodes.w.sum(1, keepdim=True).clamp_min(1e-30) * pseudo[:, None]
    total = (pseudo + real_mask.float().sum(1)).clamp_min(1e-12)
    side = real_mask.float().view(B // group, group, -1)                      # [b, group, R]
    step = _comp_step(nodes, r + extra, bank.rank)
    C = nodes.comp.shape[1]
    k = nodes.H.shape[1]

    def times(Y):
        q = Y.shape[-1]
        XY = torch.einsum("brd,bgdj->bgrj", real, Y.view(B // group, group, D, q)) * side[..., None]
        out = torch.einsum("brd,bgrj->bgdj", real, XY).reshape(B, D, q)
        for c0 in range(0, C, step):
            sl = slice(c0, c0 + step)
            dg, U = bank.parts(nodes.comp[:, sl])
            w = weight[:, sl]
            UY = torch.einsum("bcdr,bdq->bcrq", U, Y)
            out += dg.weighted(w[..., None].expand(-1, -1, q)).mT * Y
            out += torch.einsum("bcdr,bcrq->bdq", U, w[..., None, None] * UY)
            if k:
                HU = torch.einsum("bkd,bcdr->bckr", nodes.H, U)
                G = nodes.A[:, sl] @ (dg.H(nodes.H, Y) + HU @ UY)            # A H Sigma_0 Y
                out -= dg.weighted_H(nodes.H, w[..., None, None] * G).mT
                out -= torch.einsum("bcdr,bcrq->bdq", U, w[..., None, None] * (HU.mT @ G))
            m = nodes.m[:, sl]
            out += torch.einsum("bc,bcd,bcq->bdq", w, m, m @ Y)
        return out / total[:, None, None] - mean[..., None] * (mean[:, None] @ Y)

    diag = torch.einsum("bgr,brd->bgd", side, real.square()).reshape(B, D)
    for c0 in range(0, C, step):
        sl = slice(c0, c0 + step)
        dg, U = bank.parts(nodes.comp[:, sl])
        m = nodes.m[:, sl]
        own = dg.own() + U.square().sum(-1) + m.square()
        if k:
            G = dg.own()[..., None] * nodes.H.mT[:, None] + U @ torch.einsum("bkd,bcdr->bcrk", nodes.H, U)  # Sigma_0 H^T
            own = own - ((G @ nodes.A[:, sl]) * G).sum(-1)
        diag += torch.einsum("bc,bcd->bd", weight[:, sl], own)
    diag = diag / total[:, None] - mean.square()

    Y = torch.randn(B, D, r + extra, generator=generator).to(mean.device)
    for _ in range(iters):  # non-finite inputs give non-finite outputs, which the caller detects, not a crash
        Y = torch.linalg.qr(times(Y).nan_to_num(0.0, 0.0, 0.0).cpu()).Q.to(mean.device)  # MPS's QR takes seconds
    T = (Y.mT @ times(Y)).nan_to_num(0.0, 0.0, 0.0)
    vals, vecs = torch.linalg.eigh(0.5 * (T + T.mT))
    vals, vecs = vals[:, -r:].clamp_min(0), vecs[:, :, -r:]
    U = (Y @ vecs) * vals.sqrt()[:, None]
    return torch.maximum(diag - U.square().sum(-1), torch.as_tensor(floor, device=diag.device)), U


# ===== landmarks: a coarse map of the data the model reads nodes against =====

def log_Phi(alpha):
    """log Phi(alpha), stable for very negative alpha (MPS has no log_ndtr)."""
    a = alpha.clamp_max(-5.0)
    tail = -0.5 * a.square() - torch.log(-a) - 0.5 * LOG_2PI + torch.log(1 - a.pow(-2) + 3 * a.pow(-4))
    return torch.where(alpha < -5, tail, torch.log(0.5 * torch.erfc(-alpha.clamp_min(-5.0) / math.sqrt(2))))


def fit_landmarks(X, S, r, floor, generator, iters=30):
    """S landmarks of the (centered) images X [N, D]: k-means centroids (k-means++ starts), each with its
    cluster's Gaussian (top r principal directions, and the residual per-pixel variance plus floor [D]) and
    its share of the images. Returns centers [S, D], diag [S, D], U [S, D, r] and log weights [S]."""
    N, device = len(X), X.device
    sq = X.square().sum(1)

    def dist(c):  # squared distances [N, len(c)]
        return (sq[:, None] - 2 * X @ c.T + c.square().sum(1)).clamp_min(0)
    chosen = [int(torch.randint(N, (1,), generator=generator))]
    d2 = dist(X[chosen])[:, 0]
    for _ in range(S - 1):
        chosen.append(int(torch.multinomial(d2.cpu().double(), 1, generator=generator)))
        d2 = torch.minimum(d2, dist(X[chosen[-1:]])[:, 0])
    c = X[chosen].clone()
    count = torch.zeros(S, device=device)
    assign = dist(c).argmin(1)
    for _ in range(iters):
        count = torch.zeros(S, device=device).index_add_(0, assign, torch.ones(N, device=device))
        total = torch.zeros(S, D, device=device).index_add_(0, assign, X)
        c = torch.where(count[:, None] > 0, total / count.clamp_min(1)[:, None], c)
        assign = dist(c).argmin(1)
    count = torch.zeros(S, device=device).index_add_(0, assign, torch.ones(N, device=device))
    diag = floor.expand(S, D).clone()
    U = torch.zeros(S, D, r, device=device)
    for j in range(S):
        Y = (X[assign == j] - c[j]).cpu()
        if len(Y) < 2:
            continue
        _, s, Vh = torch.linalg.svd(Y, full_matrices=False)
        k = min(r, len(s))
        Uj = Vh[:k].T * (s[:k] / math.sqrt(len(Y)))
        U[j, :, :k] = Uj.to(device)
        diag[j] += (Y.square().mean(0) - Uj.square().sum(-1)).clamp_min(0).to(device)
    return c, diag, U, (count.clamp_min(1) / N).log()


def orthonormal(Z):
    """An orthonormal basis of the columns of Z [B, D, q], by two rounds of Cholesky QR: it stays on the GPU
    (copies to the CPU for its QR, thousands per walk, fill MPS's shared pool)."""
    eye = torch.eye(Z.shape[-1], device=Z.device)
    for _ in range(2):
        G = Z.mT @ Z
        L = torch.linalg.cholesky(G + 1e-6 * G.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None] * eye)
        Z = torch.linalg.solve_triangular(L, Z.mT, upper=False).mT
    return Z


# ===== the half-space model =====

class ReadBlock(nn.Module):
    """A query reads the node's memory: cross-attention (with an additive bias per memory token), then a
    feed-forward layer."""

    def __init__(self, width, heads):
        super().__init__()
        self.ln_q, self.ln_m, self.ln_f = nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, q, mem, bias):
        m = self.ln_m(mem)
        mask = bias[:, None, None].expand(-1, self.attn.num_heads, q.shape[1], -1).flatten(0, 1)
        q = q + self.attn(self.ln_q(q), m, m, attn_mask=mask, need_weights=False)[0]
        return q + self.ff(self.ln_f(q))


class ImageHead(nn.Module):
    """A feature vector -> channels of a 28 x 28 image."""

    def __init__(self, width, channels):
        super().__init__()
        self.fc = nn.Linear(width, 64 * 7 * 7)
        self.up = nn.Sequential(nn.GELU(), nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.GELU(),
                                nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(), nn.Conv2d(32, channels, 3, padding=1))

    def forward(self, f):
        lead = f.shape[:-1]
        x = self.up(self.fc(f.reshape(-1, f.shape[-1])).view(-1, 64, 7, 7))
        return x.view(*lead, x.shape[1], D)


@dataclass
class Context:
    """What queries at a batch of nodes read: the memory mem [B, L, W] (history then landmark tokens) with
    its attention bias [B, L], and the landmarks' log membership in each node [B, S]."""
    mem: Tensor
    bias: Tensor
    log_member: Tensor

    def select(self, idx):
        return Context(self.mem[idx], self.bias[idx], self.log_member[idx])


class SliceModel(nn.Module):
    """For a node (its history: oriented half-spaces H [B, k, D], o [B, k]) and query slices (qn [B, Q, D],
    qo [B, Q]): each query's side {qn . x >= qo} within the node, as the outputs asked for: "share" (its share
    of the node's mass), "projection" (the mean and variance of the distance u = qn . x - qo on the side),
    "mean" (its mean), "cov" (its covariance, diagonal plus rank r) and "levels" (256 logits per pixel for
    8-bit images on the side). A slice tree has three: the encoder (mean), the prior (share, projection) and
    a decoder, Gaussian (mean, cov) or categorical (levels).

    The memory the queries read is the encoded history and one token per landmark, carrying the landmark's
    log membership in the node: its weight times the product, over the history, of the chance its Gaussian
    falls on the node's side of each half-space. That log membership also biases the attention to it. Each
    query also gets the landmark mixture's estimate of its side (share and mean, each landmark Gaussian cut
    by the query alone)."""

    lm_center: Tensor  # [S, D] landmark Gaussians (buffers, set by set_landmarks)
    lm_diag: Tensor    # [S, D]
    lm_U: Tensor       # [D, S * r]: U [S, D, r] laid out for one matmul against normals
    lm_logw: Tensor    # [S]

    def __init__(self, rank=8, width=256, layers=4, heads=4, offset_scale=1.0, landmarks=256,
                 outputs=("share", "mean", "cov")):
        super().__init__()
        self.config = dict(rank=rank, width=width, layers=layers, heads=heads, offset_scale=offset_scale,
                           landmarks=landmarks, outputs=tuple(outputs))
        self.rank, self.offset_scale = rank, offset_scale
        self.embed = nn.Linear(D + 1, width)
        self.history_norm, self.query_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.query_images = nn.Linear(D, width)  # the landmarks' estimate of the side's mean
        self.query_scalars = nn.Linear(2, width)
        self.null = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, 0.0, "gelu", batch_first=True, norm_first=True)
        self.history = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.lm_embed = nn.Parameter(torch.randn(landmarks, width) * 0.02)
        self.lm_in = nn.Linear(2, width)
        self.lm_norm = nn.LayerNorm(width)
        self.register_buffer("lm_center", torch.zeros(landmarks, D))
        self.register_buffer("lm_diag", torch.ones(landmarks, D))
        self.register_buffer("lm_U", torch.zeros(D, landmarks * rank))
        self.register_buffer("lm_logw", torch.full((landmarks,), -math.log(landmarks)))
        self.read = nn.ModuleList(ReadBlock(width, heads) for _ in range(layers))
        self.out_norm = nn.LayerNorm(width)
        self.share_head = nn.Linear(width, 1) if "share" in outputs else None
        self.projection_head = nn.Linear(width, 2) if "projection" in outputs else None
        self.mean_head = ImageHead(width, 1) if "mean" in outputs else None
        self.cov_head = ImageHead(width, 1 + rank) if "cov" in outputs else None
        self.levels_head = ImageHead(width, 256) if "levels" in outputs else None
        self.start_basis: Tensor | None = None  # landmark_start's fixed start, made on first use

    @torch.no_grad()
    def set_landmarks(self, center, diag, U, logw):
        self.lm_center.copy_(center)
        self.lm_diag.copy_(diag)
        self.lm_U.copy_(U.permute(1, 0, 2).reshape(D, -1))
        self.lm_logw.copy_(logw)

    def landmark_terms(self, n, o):
        """Each landmark Gaussian against half-spaces n . x >= o (n [..., D], o [...]): its standardized
        distance into them alpha [..., S], its spread along n, s [..., S], and U^T n [..., S, r]."""
        Un = (n @ self.lm_U).view(*n.shape[:-1], len(self.lm_center), self.rank)
        s = (n.square() @ self.lm_diag.T + Un.square().sum(-1)).clamp_min(1e-12).sqrt()
        return (n @ self.lm_center.T - o[..., None]) / s, s, Un

    def log_membership(self, H, o):
        """The landmarks' log membership in nodes, normalized over the landmarks [B, S], and its total (the
        node's log mass under the landmark mixture) [B, 1]."""
        logm = self.lm_logw + log_Phi(self.landmark_terms(H, o)[0]).sum(-2)
        total = logm.logsumexp(-1, keepdim=True)
        return logm - total, total

    @torch.no_grad()
    def landmark_start(self, log_member, iters=3, extra=4):
        """Where Lloyd starts at nodes with landmark log membership [B, S]: the landmark mixture's mean [B, D]
        and its top r principal directions, scaled by their spread, U [B, D, r]. Subspace iteration from a
        fixed start, so a node always gets the same. The nodes are padded to a multiple of 64: MPS keeps what
        it builds for every new shape, and a walk would otherwise bring hundreds."""
        n = len(log_member)
        w = F.pad(log_member, (0, 0, 0, -n % 64)).exp()
        B, S, r = len(w), len(self.lm_center), self.rank
        mean, dg = w @ self.lm_center, w @ self.lm_diag

        def times(Y):  # the mixture's covariance times Y [B, D, q]
            q = Y.shape[-1]
            UY = (Y.mT @ self.lm_U).view(B, q, S, r) * w[:, None, :, None]
            out = dg[..., None] * Y + (UY.flatten(2) @ self.lm_U.T).mT
            out = out + self.lm_center.T @ (w[..., None] * (self.lm_center @ Y))
            return out - mean[..., None] * (mean[:, None] @ Y)
        if self.start_basis is None or self.start_basis.shape[-1] != r + extra or self.start_basis.device != w.device:
            self.start_basis = torch.randn(D, r + extra, generator=torch.Generator().manual_seed(0)).to(w.device)
        Y = self.start_basis.expand(B, -1, -1)
        for _ in range(iters):
            Y = orthonormal(times(Y))
        T = Y.mT @ times(Y)
        vals, vecs = torch.linalg.eigh(0.5 * (T + T.mT))
        return mean[:n], ((Y @ vecs[..., -r:]) * vals[:, -r:].clamp_min(0).sqrt()[:, None])[:n]

    def slice_input(self, n, o):
        """Half-spaces as vectors; the unit normal is scaled to entries of about unit size."""
        return self.embed(torch.cat([n * math.sqrt(D), (o / self.offset_scale)[..., None]], -1))

    def memory(self, H, o):
        hist = self.history(torch.cat([self.null.expand(len(H), -1, -1), self.history_norm(self.slice_input(H, o))], 1))
        rel, total = self.log_membership(H, o)
        seen = rel.clamp_min(-30)
        lm = self.lm_norm(self.lm_embed + self.lm_in(torch.stack([seen / 10, (total.clamp_min(-100) / 10).expand_as(seen)], -1)))
        bias = torch.cat([hist.new_zeros(hist.shape[:2]), seen], 1)
        return Context(torch.cat([hist, lm], 1), bias, rel)

    def features(self, ctx, qn, qo):
        alpha, s, Un = self.landmark_terms(qn, qo)                                         # [B, Q, S]
        _, lam = normal_side(alpha)
        logw = ctx.log_member[:, None] + log_Phi(alpha)
        log_share = logw.logsumexp(-1).clamp_max(-1e-6)
        w = (logw - log_share[..., None]).exp()                                           # landmarks within the side
        coef = w * lam / s
        shift = (coef @ self.lm_diag) * qn + (coef[..., None] * Un).flatten(-2) @ self.lm_U.T
        lm_mean = w @ self.lm_center + shift
        scalars = torch.stack([((qn * lm_mean).sum(-1) - qo) / self.offset_scale,
                               (log_share - torch.log(-torch.expm1(log_share))).clamp(-30, 30) / 10], -1)
        q = self.query_norm(self.slice_input(qn, qo) + self.query_images(lm_mean) + self.query_scalars(scalars))
        for block in self.read:
            q = block(q, ctx.mem, ctx.bias)
        return self.out_norm(q)

    def share_logit(self, f):
        assert self.share_head is not None
        return self.share_head(f)[..., 0]

    def projection(self, f):
        """The distance u = qn . x - qo on the query's side: its mean and variance."""
        assert self.projection_head is not None
        x = self.projection_head(f)
        return self.offset_scale * F.softplus(x[..., 0]), (self.offset_scale * F.softplus(x[..., 1])).square() + VAR_FLOOR

    def mean(self, f):
        assert self.mean_head is not None
        return self.mean_head(f)[..., 0, :]

    def levels(self, f):
        """Logits of each pixel's 256 grey levels [..., D, 256]."""
        assert self.levels_head is not None
        return self.levels_head(f).transpose(-1, -2)

    def cov(self, f):
        assert self.cov_head is not None
        x = self.cov_head(f)
        return VAR_FLOOR + 0.05 * F.softplus(x[..., 0, :] - 2), 0.1 * x[..., 1:, :].transpose(-1, -2)


# ===== Lloyd's algorithm =====

def start_slices(mu, U, seeds, n_starts):
    """Seeded starts at nodes with Gaussians (mu [B, D], low-rank part U [B, D, r]): the top principal
    directions, then random combinations of them, each through the node's mean. -> n [B, S, D], o [B, S]."""
    B, r = len(mu), U.shape[-1]
    vals, vecs = torch.linalg.eigh((U.mT @ U).cpu())
    vals, vecs = vals.flip(-1).clamp_min(1e-12), vecs.flip(-1)
    dirs = (U @ vecs.to(U.device)) / vals.sqrt().to(U.device)[:, None]        # unit principal directions
    n_pc = min(r, (n_starts + 1) // 2)
    coef = hashed_normals(seeds, (n_starts - n_pc) * r).view(B, n_starts - n_pc, r) * vals.sqrt()[:, None]
    n = torch.cat([dirs[..., :n_pc].mT, coef.to(U.device) @ dirs.mT], 1)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return n, (n * mu[:, None]).sum(-1)


def lloyd(evaluate, rank, mu, U, seeds, n_starts, iters, keep=False):
    """Lloyd's algorithm from each start: move the slice to the perpendicular bisector of its two sides'
    means, which are evaluate(qn, qo)'s second output. Then rank(n, o, out) scores the final slices (n [B,
    S, D], o [B, S], out the last evaluation): their sides' shares (+ sides, then - sides) [B, 2S] and their
    gains [B, S]; the start with the highest gain is kept. Returns the slice (n [B, D], o [B]), its sides'
    shares (+ then -) [B, 2], every iterate (n [B, S, iters + 1, D], o [B, S, iters + 1]), the chosen start
    [B], and with keep, every evaluation (evaluate's outputs at each iterate, the queries being the iterates
    then their negations)."""
    n, o = start_slices(mu, U, seeds, n_starts)
    S = n.shape[1]
    path_n, path_o, evals = [n], [o], []
    out = None
    for t in range(iters + 1):
        out = evaluate(torch.cat([n, -n], 1), torch.cat([o, -o], 1))
        mean = out[1]
        if keep:
            evals.append(out)
        if t == iters:
            break
        mid, diff = (mean[:, :S] + mean[:, S:]) / 2, mean[:, :S] - mean[:, S:]
        size = diff.norm(dim=-1, keepdim=True)
        n = torch.where(size > 1e-6, diff / size.clamp_min(1e-12), n)
        o = (n * mid).sum(-1)
        path_n.append(n)
        path_o.append(o)
    share, gain = rank(n, o, out)
    best = gain.argmax(1)
    rows = torch.arange(len(best), device=best.device)
    shares = torch.stack([share[rows, best], share[rows, best + S]], 1)
    return n[rows, best], o[rows, best], shares, torch.stack(path_n, 2), torch.stack(path_o, 2), best, evals


def exact_rank(n, o, out):
    """lloyd's rank from exact evaluations (share, mean, ...): the gain p (1 - p) (n . (mean+ - mean-))^2,
    the variance between the two sides along the normal."""
    share, mean = out[0], out[1]
    S = n.shape[1]
    p = share[:, :S] / (share[:, :S] + share[:, S:]).clamp_min(1e-12)
    return share, p * (1 - p) * (n * (mean[:, :S] - mean[:, S:])).sum(-1).square()


def model_lloyd(encoder, prior, ctx, prior_ctx):
    """lloyd's evaluate and rank on the networks, at nodes with the encoder's context ctx and the prior's
    prior_ctx: the encoder's side means, and the prior's shares and gains p (1 - p) (u+ + u-)^2, u being a
    side's mean distance from the slice (so u+ + u- = n . (mean+ - mean-))."""
    def evaluate(qn, qo):
        return None, encoder.mean(encoder.features(ctx, qn, qo))

    def rank(n, o, out):
        f = prior.features(prior_ctx, torch.cat([n, -n], 1), torch.cat([o, -o], 1))
        share, u = torch.sigmoid(prior.share_logit(f)), prior.projection(f)[0]
        S = n.shape[1]
        p = share[:, :S] / (share[:, :S] + share[:, S:]).clamp_min(1e-12)
        return share, p * (1 - p) * (u[:, :S] + u[:, S:]).square()
    return evaluate, rank


def starts_at(depth, starts, start_depths):
    for s, limit in zip(starts, start_depths):
        if depth < limit:
            return s
    return starts[-1]


# ===== deployment: the networks alone =====

@dataclass
class Root:
    """The root node's Gaussian (the training images'), and the pixel center."""
    center: Tensor  # [D] training mean, subtracted from every image
    U: Tensor       # [D, r]
    diag: Tensor    # [D]


def root_gaussian(X, r):
    """The (centered) training images' Gaussian: top r principal components plus a diagonal residual."""
    cov = X.T @ X / len(X)
    vals, vecs = torch.linalg.eigh(cov.cpu())
    U = (vecs[:, -r:] * vals[-r:].clamp_min(0).sqrt()).to(X.device)
    return U, (torch.diagonal(cov) - U.square().sum(-1)).clamp_min(VAR_FLOOR)


def empty_cache(device):
    if device.type == "mps":
        torch.mps.empty_cache()


def _nodes_of(bits):
    """The distinct codes among bits [n], each code's index for every image, and one image of each code."""
    uniq, inverse = torch.unique(bits, return_inverse=True)
    first = torch.full((len(uniq),), len(bits), dtype=torch.int64).scatter_reduce(0, inverse, torch.arange(len(bits)), "amin")
    return uniq, inverse, first


@torch.no_grad()
def walk_model(encoder, prior, K, starts, start_depths, iters, X=None, n=0, generator=None, chunk=256):
    """Encode centered images X [n, D], or (X None) sample n codes from the prior with generator. At every
    node Lloyd runs on the encoder from the landmark mixture's starts, and the prior ranks the starts and
    gives the tokens' probabilities. Returns tokens [n, K] (+1 / -1), each token's log probability [n, K],
    and each code's oriented half-spaces (H [n, K, D], o [n, K])."""
    device = encoder.lm_center.device
    n = len(X) if X is not None else n
    H = torch.zeros(n, 0, D, device=device)
    o = torch.zeros(n, 0, device=device)
    bits = torch.zeros(n, dtype=torch.int64)
    tokens = torch.zeros(n, K, dtype=torch.int8)
    logp = torch.zeros(n, K, device=device)
    for k in range(K):
        uniq, inverse, first = _nodes_of(bits)
        u = len(uniq)
        qn, qo = torch.zeros(u, D, device=device), torch.zeros(u, device=device)
        sh = torch.zeros(u, 2, device=device)
        s = starts_at(k, starts, start_depths)
        for i in range(0, u, chunk):
            rep = first[i:i + chunk].to(device)
            sl = slice(i, i + len(rep))
            ctx = encoder.memory(H[rep], o[rep])
            mu, U = encoder.landmark_start(ctx.log_member)
            evaluate, rank = model_lloyd(encoder, prior, ctx, prior.memory(H[rep], o[rep]))
            qn[sl], qo[sl], sh[sl], _, _, _, _ = lloyd(evaluate, rank, mu, U, node_seeds(uniq[sl], k), s, iters)
        inv = inverse.to(device)
        qn_i, qo_i, sh_i = qn[inv], qo[inv], sh[inv]
        p_plus = sh_i[:, 0] / sh_i.sum(1).clamp_min(1e-12)
        if X is not None:
            plus = (X * qn_i).sum(1) >= qo_i
        else:
            plus = torch.rand(n, generator=generator).to(device) < p_plus
        a = torch.where(plus, 1.0, -1.0)
        logp[:, k] = torch.where(plus, p_plus, 1 - p_plus).clamp_min(1e-12).log()
        tokens[:, k] = a.cpu().to(torch.int8)
        bits = bits | (plus.cpu().to(torch.int64) << k)
        H = torch.cat([H, (qn_i * a[:, None])[:, None]], 1)
        o = torch.cat([o, (qo_i * a)[:, None]], 1)
        empty_cache(device)
    return tokens, logp, H, o


@torch.no_grad()
def decode(decoder, root, tokens, H, o, X=None, chunk=256):
    """The decoder's Gaussians at the nodes along codes (tokens [n, K], their oriented half-spaces H, o):
    each prefix's mean [n, K + 1, D] (the root's is 0), the leaves' (diag [n, D], U [n, D, r]), and with
    centered images X [n, D], each prefix's log density of them [n, K + 1]."""
    device = H.device
    n, K = tokens.shape
    mean = torch.zeros(n, D, device=device)
    diag, U = root.diag.expand(n, -1), root.U.expand(n, -1, -1)
    means, logp = [mean], []
    if X is not None:
        logp.append(lowrank_logpdf(X, mean, diag, U))
    bits = torch.zeros(n, dtype=torch.int64)
    for k in range(K):
        uniq, inverse, first = _nodes_of(bits)
        u = len(uniq)
        side_mean = torch.zeros(u, 2, D, device=device)
        side_diag = torch.zeros(u, 2, D, device=device)
        side_U = torch.zeros(u, 2, D, decoder.rank, device=device)
        for i in range(0, u, chunk):
            rep = first[i:i + chunk]
            sl = slice(i, i + len(rep))
            a = tokens[rep, k].float().to(device)
            rep = rep.to(device)
            qn, qo = H[rep, k] * a[:, None], o[rep, k] * a                         # the slice, toward its + side
            f = decoder.features(decoder.memory(H[rep, :k], o[rep, :k]), torch.stack([qn, -qn], 1), torch.stack([qo, -qo], 1))
            side_mean[sl] = decoder.mean(f)
            side_diag[sl], side_U[sl] = decoder.cov(f)
        plus = tokens[:, k] > 0
        inv, side = inverse.to(device), (~plus).long().to(device)
        mean, diag, U = side_mean[inv, side], side_diag[inv, side], side_U[inv, side]
        means.append(mean)
        if X is not None:
            logp.append(lowrank_logpdf(X, mean, diag, U))
        bits = bits | (plus.to(torch.int64) << k)
        empty_cache(device)
    return torch.stack(means, 1), (diag, U), (torch.stack(logp, 1) if X is not None else None)


def shallow_loss(tokens, X, k_max=8):
    """Per depth k <= k_max: the held-out decompression loss estimated within nodes (mean distance between
    two different held-out images of a node, weighted by its images), and the share of images alone in
    their node, which the estimate cannot see."""
    out = []
    for k in range(1, min(k_max, tokens.shape[1]) + 1):
        bits = ((tokens[:, :k] > 0).to(torch.int64) << torch.arange(k)).sum(1)
        uniq, inv, counts = torch.unique(bits, return_inverse=True, return_counts=True)
        inv = inv.to(X.device)
        s1 = torch.zeros(len(uniq), D, device=X.device).index_add_(0, inv, X)
        s2 = torch.zeros(len(uniq), device=X.device).index_add_(0, inv, X.square().sum(1))
        c = counts.to(X.device).float()
        pair = 2 * (c * s2 - s1.square().sum(1)) / (c * (c - 1)).clamp_min(1) / D
        ok = c >= 2
        out.append(((pair * c)[ok].sum() / c[ok].sum()).item() if ok.any() else math.nan)
        out[-1] = (out[-1], 1 - c[ok].sum().item() / len(X))
    return out
