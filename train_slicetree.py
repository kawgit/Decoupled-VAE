"""Train a slice tree (see README): round 0's density, then DAgger passes of the encoder and the prior, then
the decoder on the final tree.

Three half-space models share nothing. The encoder predicts side means, which Lloyd's algorithm iterates
on. The prior predicts each side's share and the mean and variance of its distance u from the slice: the
gain p (1 - p) (u+ + u-)^2 ranks Lloyd's starts, and the shares give the tokens' probabilities. The decoder
predicts side means and covariances, for decompression and the leaves' Gaussians. Lloyd starts from the
landmark mixture inside the node (its mean and principal directions), so the tree is the encoder's and the
prior's alone.

Each pass walks the tree along the paths of a sample of training images. At every node it runs Lloyd, on
the exact combined targets in the first pass and on the encoder and prior in later ones, and labels with
exact targets: for the encoder, Lloyd's queries (one iterate per start, and the chosen slice, each from both
sides) with their sides' means, and the node's Gaussian; for the prior, every start's final slice from both
sides; for the decoder, the chosen slice's two sides with their covariances (which are also the children's
Gaussians). The encoder is trained by the whitened bisector loss: the squared distance, in the node's own
whitened coordinates, between the slice Lloyd moves to on the exact side means and the one it moves to on
the encoder's. A split's share of the log decompression loss at every depth below it weights it. The prior
is trained by the share's cross-entropy plus, weighted by the share, the distance's Gaussian log-loss.

Each stops early on its own held-out score. The encoder's comes from one walk of held-out images through
the current tree per epoch: their log decompression loss against the mean of the training images in their
node (depths up to --enc-val-depth, weighted by rho^k). The prior's is its own loss on held-out queries
labelled exactly on the tree its pass's labels come from (the exact tree in the first pass, after that the
previous pass's): its token bits on the current tree would rise whenever the encoder's splits improve. Once
per pass, those held-out images are walked through the new tree and labelled, as a diagnostic and as the
next pass's prior validation; gamma is refit from that walk.
After the last pass, --decoder-walks walks of the final tree label its chosen slices, and the decoder is
trained on them, stopping early on the held-out images' log likelihood under its node Gaussians (depths
weighted by rho^k).
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")  # ~6.5 GB: fail instead of swapping
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dvae.data import load_mnist  # noqa: E402
from dvae.slicetree import (D, ELEMENTS, Nodes, Root, SliceModel, bernoulli_kl, bits_per_dim,  # noqa: E402
                            combined_sides, cut, decode, dequantize, exact_rank, fit_landmarks, fit_parzen,
                            gaussian_ce, gaussian_entropy, lloyd, mixture_logpdf, model_lloyd, nearest,
                            node_seeds, root_gaussian, root_nodes, shallow_loss, side_covariance, starts_at,
                            walk_model)

ROLES = ("enc", "prior", "dec")


def empty_cache(device):
    if device.type == "mps":
        torch.mps.empty_cache()


def memory(device):
    if device.type != "mps":
        return ""
    return f", {torch.mps.current_allocated_memory() / 2**30:.2f} GB in use, {torch.mps.driver_allocated_memory() / 2**30:.2f} GB held"


def get_device(name):
    if name:
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Table:
    """Every node of one walk: the oriented half-space that made it, and its parent (node 0 is the root)."""

    def __init__(self):
        self.parts = [(torch.zeros(1, D, dtype=torch.half), torch.zeros(1), torch.zeros(1, dtype=torch.long))]
        self.count = 1
        self.n, self.o, self.parent = torch.zeros(0), torch.zeros(0), torch.zeros(0, dtype=torch.long)

    def add(self, n, o, parent):
        ids = torch.arange(self.count, self.count + len(n))
        self.parts.append((n.half().cpu(), o.float().cpu(), parent.cpu()))
        self.count += len(n)
        return ids

    def finish(self):
        self.n, self.o, self.parent = (torch.cat(t) for t in zip(*self.parts))
        self.parts = []

    def to(self, device):
        table = Table()
        table.parts, table.count = [], self.count
        table.n, table.o, table.parent = self.n.to(device), self.o.to(device), self.parent.to(device)
        return table

    def history(self, ids, k):
        """The half-spaces along each node's code: H [B, k, D], o [B, k] (on the table's device)."""
        path = []
        cur = ids
        for _ in range(k):
            path.append(cur)
            cur = self.parent[cur]
        path = torch.stack(path, 1) if path else torch.zeros(len(ids), 0, dtype=torch.long, device=ids.device)
        return self.n[path].float(), self.o[path]


class LabelStore:
    """One walk's labels, as a sink for walk: per depth, "node" labels per node, and each network's ("enc",
    "prior", "dec") per query, each node's queries together and the nodes in one order. Each depth goes to
    disk as soon as it is done and is read back memory-mapped, so a walk's labels never sit in memory."""

    def __init__(self, directory):
        self.dir = directory
        os.makedirs(directory, exist_ok=True)
        self.table = Table()
        self.pending: dict[str, list[dict]] = {}
        self.depths: dict[int, dict[str, dict]] = {}

    def __call__(self, k, H, o, labels):
        for role, parts in labels.items():
            self.pending.setdefault(role, []).append(
                {key: ((v.half() if v.dim() >= 2 else v.float()) if v.is_floating_point() else v).cpu()
                 for key, v in parts.items()})

    def flush(self, k):
        if self.pending:
            torch.save({role: {key: torch.cat([p[key] for p in parts]) for key in parts[0]}
                        for role, parts in self.pending.items()}, os.path.join(self.dir, f"depth{k:02d}.pt"))
            self.pending = {}

    def finish(self, key=None):
        self.table.finish()
        torch.save(dict(n=self.table.n, o=self.table.o, parent=self.table.parent), os.path.join(self.dir, "table.pt"))
        if key is not None:
            with open(os.path.join(self.dir, "key.json"), "w") as f:
                json.dump(key, f)
        self.load()

    def load(self):
        table = torch.load(os.path.join(self.dir, "table.pt"))
        self.table = Table()
        self.table.parts, self.table.n, self.table.o, self.table.parent = [], table["n"], table["o"], table["parent"]
        self.depths = {int(f[5:7]): torch.load(os.path.join(self.dir, f), mmap=True)
                       for f in sorted(os.listdir(self.dir)) if f.startswith("depth")}

    @staticmethod
    def cached(directory, key):
        """The store saved in directory, if its settings match key."""
        try:
            with open(os.path.join(directory, "key.json")) as f:
                if json.load(f) != key:
                    return None
        except (OSError, ValueError):
            return None
        store = LabelStore(directory)
        store.load()
        return store

    @staticmethod
    def fresh(directory):
        if os.path.isdir(directory):
            for f in os.listdir(directory):
                os.remove(os.path.join(directory, f))
        return LabelStore(directory)

    def size(self):
        counts = {}
        for part in self.depths.values():
            for role, labels in part.items():
                counts[role] = counts.get(role, 0) + len(next(iter(labels.values())))
        return ", ".join(f"{n} {role}" for role, n in counts.items())


@dataclass
class Level:
    ids: torch.Tensor     # [B] node ids in the walk's table
    bits: torch.Tensor    # [B] int64 codes
    nodes: Nodes          # the round's density inside each node
    real: list            # training images in each node
    walkers: list         # the walked images in each node (indices into the walk's images)
    pbar: torch.Tensor    # [B] combined mass of the node
    m: torch.Tensor       # [B, D] the node's exact Gaussian: mean,
    diag: torch.Tensor    # [B, D] and covariance, diagonal plus low rank
    U: torch.Tensor       # [B, D, r]


def pad_images(X, lists):
    R = max(1, max(len(x) for x in lists))
    idx = torch.zeros(len(lists), R, dtype=torch.long)
    mask = torch.zeros(len(lists), R, dtype=torch.bool)
    for i, x in enumerate(lists):
        idx[i, :len(x)] = x
        mask[i, :len(x)] = True
    return X[idx.to(X.device)], mask.to(X.device)


def split(X, lists, qn, qo):
    """Each node's images on the + side of its slice, and on the - side."""
    plus, minus = [], []
    for i, x in enumerate(lists):
        if len(x) == 0:
            plus.append(x)
            minus.append(x)
            continue
        side = ((X[x.to(X.device)] @ qn[i]) >= qo[i]).cpu()
        plus.append(x[side])
        minus.append(x[~side])
    return plus, minus


def chunks(level, max_nodes=256, max_rows=2**24 // D):
    """Node index groups, by decreasing number of real images, each padding to at most max_rows images."""
    order = sorted(range(len(level.real)), key=lambda i: -len(level.real[i]))
    out, cur, width = [], [], 0
    for i in order:
        width = max(width, len(level.real[i]), 1)
        if cur and (len(cur) >= max_nodes or width * (len(cur) + 1) > max_rows):
            out.append(cur)
            cur, width = [], max(len(level.real[i]), 1)
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def finite(*ts):
    """Whether each query's targets are all finite: ts [b, Q, ...] -> [b, Q]."""
    good = torch.ones(ts[0].shape[:2], dtype=torch.bool, device=ts[0].device)
    for t in ts:
        good &= torch.isfinite(t).flatten(2).all(-1) if t.dim() > 2 else torch.isfinite(t)
    return good


def walk(args, bank, root, X, W, nets, teacher, gamma, gen, log, sink, budget, roles=ROLES):
    """Walk the tree along the paths of images W [n, D] (centered), from the root to depth K. At every node
    they reach, run Lloyd from the landmark starts (on the exact targets when teacher, else on nets, the
    encoder and the prior), label queries with exact targets, hand them to sink(k, H, o, labels), then cut
    the node by its chosen slice. labels holds "node" (ids, weight: the node's mass over the chance the walk
    reached it, and its exact Gaussian m, diag, U) and, for each role asked for, its queries: "enc" (Lloyd's
    queries: qn, qo, share, mean), "prior" (every start's final slice: qn, qo, share, u_mean, u_var) and "dec"
    (the chosen slice: qn, qo, share, mean, diag, U), each with ok (0 where the targets are not finite). Each
    role's queries are a set of slices and then the same slices from their other side. X [N, D] are the real
    (training) images of the combined targets; budget caps the density components held per depth. Returns,
    for fitting gamma, each walked image's (density share, real images in the node, real images on its side)
    at every node it passes."""
    encoder, prior = nets
    device, N, r = X.device, len(X), bank.rank
    floor = bank.kernel_diag
    records = []
    bad_count = 0
    level = Level(torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.int64), root_nodes(bank, device),
                  [torch.arange(N)], [torch.arange(len(W))], torch.ones(1), torch.zeros(1, D, device=device),
                  root.diag[None], root.U[None])
    table = getattr(sink, "table", None) or Table()
    t0 = time.time()

    def inclusion(nz):  # chance that a node holding nz training images is reached by the walked images
        return 1 - (1 - nz / N).clamp(0, 1).pow(len(W))

    for k in range(args.K):
        B = len(level.ids)
        # components per node: the depth's budget, and few enough that their cut state (a mean and a k x k
        # matrix each, for the nodes and their children) stays within --comp-floats
        k_cut = k + 1
        cap = max(args.min_comps, min(budget // (2 * B), args.comp_floats // (2 * B * (D + k_cut * k_cut))))
        S = starts_at(k, args.starts, args.start_depths)
        parts = []
        for idx in chunks(level):
            idx_t = torch.tensor(idx)
            idx_d = idx_t.to(device)
            nodes = level.nodes.select(idx_d)
            real, mask = pad_images(X, [level.real[i] for i in idx])
            nz = mask.float().sum(1)
            alpha = gamma * (N - nz)
            ids = level.ids[idx_t]
            b = len(idx)

            def exact(qn, qo, nodes=nodes, real=real, mask=mask, alpha=alpha):
                return combined_sides(bank, nodes, real, mask, alpha, qn, qo)

            if teacher:
                mu, U0 = encoder.landmark_start(encoder.log_membership(nodes.H, nodes.o)[0])
                evaluate, rank = exact, exact_rank
            else:
                ctx = encoder.memory(nodes.H, nodes.o)
                mu, U0 = encoder.landmark_start(ctx.log_member)
                evaluate, rank = model_lloyd(encoder, prior, ctx, prior.memory(nodes.H, nodes.o))
            qn, qo, _, pn, po, best, evals = lloyd(evaluate, rank, mu, U0, node_seeds(level.bits[idx_t], k), S,
                                                     args.iters, keep=teacher)

            # the encoder's queries: one iterate per start at a uniform iteration, then the chosen slice; the
            # prior's: every start's final slice; the decoder's: the chosen slice. Each then from its other side.
            pick = torch.randint(0, args.iters + 1, (b, S), generator=gen).to(device)
            rows = torch.arange(b, device=device)[:, None]
            cols = torch.arange(S, device=device)[None]
            En = torch.cat([pn[rows, cols, pick], qn[:, None]], 1)
            Eo = torch.cat([po[rows, cols, pick], qo[:, None]], 1)
            En, Eo = torch.cat([En, -En], 1), torch.cat([Eo, -Eo], 1)
            Pn, Po = torch.cat([pn[:, :, -1], -pn[:, :, -1]], 1), torch.cat([po[:, :, -1], -po[:, :, -1]], 1)
            Dn, Do = torch.stack([qn, -qn], 1), torch.stack([qo, -qo], 1)
            enc_t = prior_t = None
            if teacher:  # Lloyd evaluated all of these exactly already: the iterates, then their negations
                paths = [torch.stack([e[j] for e in evals], 2) for j in range(6)]  # [b, 2S, iters + 1, ...]

                def at(path, starts, step):
                    return torch.cat([path[rows, starts, step], path[rows, starts + S, step]], 1)
                picked = [at(pth, cols.expand(b, -1), pick) for pth in paths]
                dec_t = [at(pth, best[:, None], torch.full_like(best[:, None], args.iters)) for pth in paths]
                enc_t = [torch.cat([pk[:, :S], fn[:, :1], pk[:, S:], fn[:, 1:]], 1) for pk, fn in zip(picked, dec_t)]
                prior_t = [pth[:, :, args.iters] for pth in paths]
            elif "enc" in roles or "prior" in roles:
                QE = En.shape[1]
                both = combined_sides(bank, nodes, real, mask, alpha, torch.cat([En, Pn], 1), torch.cat([Eo, Po], 1))
                enc_t, prior_t = [t[:, :QE] for t in both], [t[:, QE:] for t in both]
                dec_t = [torch.cat([t[:, S:S + 1], t[:, 2 * S + 1:2 * S + 2]], 1) for t in enc_t]
            else:
                dec_t = list(combined_sides(bank, nodes, real, mask, alpha, Dn, Do))
            share_d, mean_d, density_d, nside_d = dec_t[:4]

            # the chosen slice's two sides: their combined covariances
            diag_d = torch.empty(b, 2, D, device=device)
            U_d = torch.empty(b, 2, D, r, device=device)
            side_real = (torch.einsum("brd,bqd->brq", real, Dn) >= Do[:, None]) & mask[..., None]
            per = max(1, 2**15 // nodes.comp.numel())  # queries at once: the copied components fill about 2^15 slots
            for q0 in range(0, 2, per):
                q1 = min(2, q0 + per)
                g = q1 - q0
                rep = torch.arange(b, device=device).repeat_interleave(g)
                side = cut(bank, nodes.select(rep), Dn[:, q0:q1].flatten(0, 1), Do[:, q0:q1].flatten(), nodes.comp.shape[1])
                d, u = side_covariance(bank, side, (alpha[:, None] * density_d[:, q0:q1]).flatten(), real,
                                       side_real[..., q0:q1].transpose(1, 2).flatten(0, 1), mean_d[:, q0:q1].flatten(0, 1),
                                       r, gen, floor=floor, group=g)
                diag_d[:, q0:q1], U_d[:, q0:q1] = d.view(b, g, D), u.view(b, g, D, r)
                del side, d, u
                if nodes.comp.numel() > 2**14:
                    empty_cache(device)

            # a side with (almost) no combined mass has no meaningful targets: ok 0, and finite stand-ins
            dec_ok = finite(share_d, mean_d, diag_d, U_d)
            bad_count += int((~dec_ok).sum())
            share_d = torch.where(dec_ok, share_d, 0.5)
            mean_d = torch.where(dec_ok[..., None], mean_d, 0.0)
            diag_d = torch.where(dec_ok[..., None], diag_d, floor)
            U_d = torch.where(dec_ok[..., None, None], U_d, 0.0)
            labels = {"node": dict(ids=ids, weight=(level.pbar[idx_t] / inclusion(nz.cpu()).clamp_min(1e-12)).to(device),
                                   m=level.m[idx_d], diag=level.diag[idx_d], U=level.U[idx_d])}
            if "enc" in roles and enc_t is not None:
                share, mean = enc_t[0], enc_t[1]
                good = finite(share, mean)
                bad_count += int((~good).sum())
                labels["enc"] = dict(qn=En.flatten(0, 1), qo=Eo.flatten(), share=torch.where(good, share, 0.5).flatten(),
                                     mean=torch.where(good[..., None], mean, 0.0).flatten(0, 1), ok=good.float().flatten())
            if "prior" in roles and prior_t is not None:
                share, u_mean, u_var = prior_t[0], prior_t[4], prior_t[5]
                good = finite(share, u_mean, u_var)
                bad_count += int((~good).sum())
                labels["prior"] = dict(qn=Pn.flatten(0, 1), qo=Po.flatten(), share=torch.where(good, share, 0.5).flatten(),
                                       u_mean=torch.where(good, u_mean, 1.0).flatten(),
                                       u_var=torch.where(good, u_var, 1.0).flatten(), ok=good.float().flatten())
            if "dec" in roles:
                labels["dec"] = dict(qn=Dn.flatten(0, 1), qo=Do.flatten(), share=share_d.flatten(),
                                     mean=mean_d.flatten(0, 1), diag=diag_d.flatten(0, 1), U=U_d.flatten(0, 1),
                                     ok=dec_ok.float().flatten())
            sink(k, nodes.H, nodes.o, labels)

            # the chosen slice's two sides: the children
            real_sides = split(X, [level.real[i] for i in idx], qn, qo)
            walk_sides = split(W, [level.walkers[i] for i in idx], qn, qo)
            for side, a in ((0, 1.0), (1, -1.0)):
                for i in range(b):
                    nw = len(walk_sides[side][i])
                    if nw:
                        records.append(torch.tensor([[density_d[i, side].item(), nz[i].item(), nside_d[i, side].item()]]).expand(nw, 3))
                keep = [i for i in range(b) if len(walk_sides[side][i])]
                if not keep or k + 1 == args.K:
                    continue
                kt = torch.tensor(keep)
                kd = kt.to(device)
                sn, so = a * qn[kd], a * qo[kd]
                c_real = [real_sides[side][i] for i in keep]
                index, present = pad_images(torch.arange(N, device=device)[:, None], c_real)
                protect = torch.where(present, index[..., 0], -1)  # the real images' own kernels
                child = cut(bank, nodes.select(kd), sn, so, cap, protect)
                child_ids = table.add(sn, so, ids[kt])
                pbar = level.pbar[idx_t][kt] * (share_d[kd, side] * dec_ok[kd, side]).cpu()  # no target: no weight
                bits = level.bits[idx_t][kt] | (int(a > 0) << k)
                parts.append(Level(child_ids, bits, child, c_real, [walk_sides[side][i] for i in keep], pbar,
                                   mean_d[kd, side], diag_d[kd, side], U_d[kd, side]))
            if B > 256:  # several chunks a depth: return their cached buffers, or MPS's pool fills up
                empty_cache(device)
        if hasattr(sink, "flush"):
            sink.flush(k)
        if bad_count:
            log(f"  depth {k:2d}: {bad_count} queries so far had non-finite targets (weighted 0)")
        done = not parts or k + 1 == args.K
        log(f"  depth {k:2d}: {B:5d} nodes, {cap:6d} components cap ({time.time() - t0:.0f}s{memory(device)})")
        if done:
            break
        level = Level(torch.cat([p.ids for p in parts]), torch.cat([p.bits for p in parts]),
                      Nodes.cat([p.nodes for p in parts]), sum((p.real for p in parts), []),
                      sum((p.walkers for p in parts), []), torch.cat([p.pbar for p in parts]),
                      torch.cat([p.m for p in parts]), torch.cat([p.diag for p in parts]), torch.cat([p.U for p in parts]))
        del parts
        empty_cache(device)
    return torch.cat(records) if records else torch.zeros(0, 3)


def fit_gamma(records, N):
    """Empirical Bayes: the gamma under which held-out images' tokens are most likely."""
    if len(records) == 0:
        return None
    density, nz, nside = records.double().unbind(1)
    best = (-math.inf, 1e-3)
    for g in torch.logspace(-7, 0, 36).tolist():
        alpha = g * (N - nz)
        p = (alpha * density + nside) / (alpha + nz).clamp_min(1e-30)
        best = max(best, (p.clamp_min(1e-12).log().mean().item(), g))
    return best[1]


class Metrics:
    """Weighted sums of named per-query terms, by depth quarter."""

    def __init__(self, names, device, K, bands=4):
        self.names = names
        self.sums = torch.zeros(len(names), bands, device=device)
        self.counts = torch.zeros(len(names), bands, device=device)
        self.K, self.bands = K, bands

    @torch.no_grad()
    def add(self, k, terms, weights):
        band = min(self.bands - 1, k * self.bands // self.K)
        for i, (values, w) in enumerate(zip(terms, weights)):
            self.sums[i, band] += (w * values).sum()
            self.counts[i, band] += w.sum()

    def total(self, name):
        i = self.names.index(name)
        return (self.sums[i].sum() / self.counts[i].sum().clamp_min(1e-30)).item()

    def text(self):
        avg = (self.sums / self.counts.clamp_min(1e-30)).tolist()
        total = (self.sums.sum(1) / self.counts.sum(1).clamp_min(1e-30)).tolist()
        seen = (self.counts > 0).tolist()
        return "; ".join(f"{n} {total[i]:.4f} (" + " ".join(f"{v:.3f}" if s else "-" for v, s in zip(avg[i], seen[i])) + ")"
                         for i, n in enumerate(self.names))


# ===== the three networks' losses: on a batch of nodes (histories H [b, k, D], o [b, k]; their node labels
# node, each [b, ...]; their queries q, each [b, Q, ...]; a weight per node [b]), the weighted sum over nodes =====

def bisector_loss(plus, minus, t_plus, t_minus, m, diag, U):
    """The squared distance between the slice Lloyd moves to on predicted side means (plus, minus [b, P, D])
    and the one it moves to on exact ones (t_plus, t_minus), in coordinates whitened by the node's Gaussian
    (m [b, D], covariance diag [b, D] + U U^T, U [b, D, r]): 2 (1 - cos) of the angle between the whitened
    normals, plus the squared difference of the standardized offsets (n . m - o) / sqrt(n^T Sigma n). Returns
    it and its two parts, [b, P] each."""
    def quad(a, c):  # a^T Sigma c
        return (a * c * diag[:, None]).sum(-1) + ((a @ U) * (c @ U)).sum(-1)
    d, t = plus - minus, t_plus - t_minus
    qd, qt = quad(d, d).clamp_min(1e-12), quad(t, t).clamp_min(1e-12)
    angle = 2 * (1 - quad(d, t) / (qd * qt).sqrt())
    a_hat = (d * (m[:, None] - (plus + minus) / 2)).sum(-1) / qd.sqrt()
    a_t = (t * (m[:, None] - (t_plus + t_minus) / 2)).sum(-1) / qt.sqrt()
    offset = (a_hat - a_t).square()
    return angle + offset, angle, offset


def encoder_loss(model, k, H, o, node, q, weight, metrics):
    """The whitened bisector loss on each labelled slice (its two sides are queries j and j + P)."""
    mean = model.mean(model.features(model.memory(H, o), q["qn"], q["qo"]))
    P = mean.shape[1] // 2
    t_mean, share, ok = q["mean"], q["share"], q["ok"]
    loss, angle, offset = bisector_loss(mean[:, :P], mean[:, P:], t_mean[:, :P], t_mean[:, P:], node["m"], node["diag"], node["U"])
    good = (ok[:, :P] * ok[:, P:] * (share[:, :P] > 1e-4) * (share[:, P:] > 1e-4)
            * ((t_mean[:, :P] - t_mean[:, P:]).square().sum(-1) > 1e-8))  # a bisector exists
    w = weight[:, None] * good / P
    metrics.add(k, (loss, angle, offset), (w, w, w))
    return (w * loss).sum() / len(H)


def prior_loss(model, k, H, o, node, q, weight, metrics):
    """The share's cross-entropy plus, weighted by the share, the Gaussian log-loss of the distance u."""
    f = model.features(model.memory(H, o), q["qn"], q["qo"])
    logit, (u, v) = model.share_logit(f), model.projection(f)
    p, t_u, t_v = q["share"].clamp(0, 1), q["u_mean"], q["u_var"]
    ce = 0.5 * (math.log(2 * math.pi) + v.log() + (t_v + (t_u - u).square()) / v)
    w = weight[:, None] * q["ok"] / q["qo"].shape[1]
    with torch.no_grad():
        kl_share = bernoulli_kl(p, logit)
        kl_u = ce - 0.5 * (math.log(2 * math.pi * math.e) + t_v.log())
        metrics.add(k, (kl_share, kl_u, kl_share + p * kl_u), (w, w * p, w))
    return (w * (F.binary_cross_entropy_with_logits(logit, p, reduction="none") + p * ce)).sum() / len(H)


def decoder_loss(model, k, H, o, node, q, weight, metrics):
    """The share times the side's Gaussian log-loss."""
    f = model.features(model.memory(H, o), q["qn"], q["qo"])
    mean, (diag, U) = model.mean(f), model.cov(f)
    p = q["share"].clamp(0, 1)
    t_mean, t_diag, t_U = q["mean"], q["diag"], q["U"]
    ce = gaussian_ce(t_mean, t_diag, t_U, mean, diag, U)
    w = weight[:, None] * q["ok"] * p / q["qo"].shape[1]
    with torch.no_grad():
        rel = (mean - t_mean).square().sum(-1) / (t_diag.sum(-1) + t_U.square().sum((-2, -1)))
        metrics.add(k, (rel, ce - gaussian_entropy(t_diag, t_U)), (w, w))
    return (w * ce).sum() / len(H)


METRICS = {encoder_loss: ("bisector", "angle", "offset"), prior_loss: ("share KL", "u KL", "objective"),
           decoder_loss: ("mean", "gauss KL")}


@dataclass
class Net:
    model: SliceModel
    opt: torch.optim.Optimizer
    role: str        # the labels it learns from
    loss: object     # one of the losses above
    depth: object    # k -> the weight of depth k's nodes

    def compute(self, k, H, o, node, q, weight, metrics):
        assert callable(self.loss)
        return self.loss(self.model, k, H, o, node, q, weight, metrics)


def rows(labels, chosen, Q, device):
    """The labels of the queries of nodes chosen, as [b, Q, ...] floats."""
    sel = (chosen[:, None] * Q + torch.arange(Q)).flatten()
    return {key: v[sel].to(device).float().view(len(chosen), Q, *v.shape[1:]) for key, v in labels.items()}


def fit(nets, stores, args, device, score, log, save=None):
    """Epochs over the labels of stores, for every network of nets ({name: Net}), in batches of nodes of one
    depth (about args.batch queries), so each node's history is encoded once. Batches are drawn with each
    node's chance proportional to its weight (its own weight times its network's depth weight), and the loss
    averages over the batch's nodes unweighted: the same objective as weighting the loss, but with the
    weight in how often a node is seen. (Weighting the loss instead puts the heavy shallow nodes into a few
    huge batches, which gradient clipping then cuts down to ordinary steps.) An epoch draws as many batches
    as the labels fill. After each epoch, score() gives each network's held-out score (lower is better) and a
    line to log; a network stops once its score has not improved for args.patience epochs, and goes back to
    its best epoch."""
    tables = {id(store): store.table.to(device) for store in stores}
    jobs: dict[str, list] = {name: [] for name in nets}
    for name, net in nets.items():
        assert callable(net.depth)
        for store in stores:
            for k, part in store.depths.items():
                if net.role not in part:
                    continue
                n_nodes = len(part["node"]["ids"])
                Q = len(part[net.role]["qo"]) // n_nodes
                jobs[name].append((store, k, part, Q, part["node"]["weight"].float() * net.depth(k)))
    best: dict[str, tuple[float, int, dict | None]] = {name: (math.inf, -1, None) for name in nets}  # score, epoch, state
    active = set(nets)
    for epoch in range(args.epochs):
        if not active:
            break
        t0 = time.time()
        batches = []
        for name in sorted(active):
            mass = torch.tensor([job[4].sum().item() for job in jobs[name]])
            count = sum(math.ceil(len(job[4]) / max(1, args.batch // job[3])) for job in jobs[name])
            for j in torch.multinomial(mass / mass.sum(), count, replacement=True).tolist():
                store, k, part, Q, weight = jobs[name][j]
                chosen = torch.multinomial(weight, max(1, args.batch // Q), replacement=True).sort().values
                batches.append((name, store, k, part, Q, chosen))
        metrics = {name: Metrics(METRICS[nets[name].loss], device, args.K) for name in active}
        for net in nets.values():
            net.model.train()
        for step, i in enumerate(torch.randperm(len(batches)).tolist()):
            name, store, k, part, Q, chosen = batches[i]
            net = nets[name]
            assert callable(net.depth)
            node = {key: v[chosen].to(device).float() for key, v in part["node"].items() if key != "ids"}
            H, o = tables[id(store)].history(part["node"]["ids"][chosen].to(device), k)
            loss = net.compute(k, H, o, node, rows(part[net.role], chosen, Q, device),
                               torch.ones(len(chosen), device=device), metrics[name])
            net.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.model.parameters(), 1.0)
            net.opt.step()
            if step % 200 == 199:
                empty_cache(device)
        t1 = time.time()
        for name in sorted(active):
            log(f"  epoch {epoch:2d} {name} train: {metrics[name].text()}")
        for net in nets.values():
            net.model.eval()
        scores, text = score()
        log(f"  epoch {epoch:2d} val: {text} ({t1 - t0:.0f}s train, {time.time() - t1:.0f}s val)")
        improved = False
        for name in sorted(active):
            value, at, state = best[name]
            if scores[name] < value:
                best[name] = (scores[name], epoch, {key: v.detach().clone() for key, v in nets[name].model.state_dict().items()})
                improved = True
            elif epoch - at >= args.patience and state is not None:
                active.discard(name)
                nets[name].model.load_state_dict(state)
                log(f"  {name}: kept epoch {at} ({value:.5f})")
        if improved and save is not None:
            current = {name: {key: v.detach().clone() for key, v in net.model.state_dict().items()} for name, net in nets.items()}
            for name, net in nets.items():
                state = best[name][2]
                if state is not None:
                    net.model.load_state_dict(state)
            save()
            for name, net in nets.items():
                net.model.load_state_dict(current[name])
    for name in sorted(active):
        value, at, state = best[name]
        if state is not None:
            nets[name].model.load_state_dict(state)
            log(f"  {name}: kept epoch {at} ({value:.5f})")


def held_out_labels(nets, args, bank, root, X, Wval, gamma, log, teacher=False, roles=("enc", "prior")):
    """Walk the held-out images Wval through the tree (exact Lloyd when teacher, else the encoder's and the
    prior's) and label the queries there exactly: a list of each chunk's (k, H, o, labels), on the CPU, and
    the walk's records for gamma. The walk's component budget scales with its images, so a deep node holds
    as many components as in training."""
    out = []

    def sink(k, H, o, labels):
        out.append((k, H.cpu(), o.cpu(), {role: {key: v.cpu() for key, v in parts.items()} for role, parts in labels.items()}))

    with torch.no_grad():
        records = walk(args, bank, root, X, Wval, (nets["encoder"].model, nets["prior"].model), teacher, gamma,
                       torch.Generator().manual_seed(args.seed + 1),
                       lambda m: log("  held-out walk:" + m) if "non-finite" in m else None, sink,
                       max(1, args.comp_budget * args.val_walkers // args.samples), roles=roles)
    return out, records


@torch.no_grad()
def held_out_metrics(nets, labelled, args, device):
    """Each network's loss (the prior's in its KL form) averaged over labelled held-out queries."""
    metrics = {name: Metrics(METRICS[net.loss], device, args.K) for name, net in nets.items()}
    for k, H, o, labels in labelled:
        H, o = H.to(device), o.to(device)
        node = {key: v.to(device).float() for key, v in labels["node"].items() if key != "ids"}
        for name, net in nets.items():
            if net.role not in labels:
                continue
            assert callable(net.depth)
            q = {key: v.to(device).float().view(len(H), -1, *v.shape[1:]) for key, v in labels[net.role].items()}
            net.compute(k, H, o, node, q, node["weight"] * net.depth(k), metrics[name])
    return metrics


@torch.no_grad()
def train_estimate(X, Xs, tokens, H, o, k_max):
    """Each held-out image's squared error per pixel against the mean of the training images X in its node,
    at depths 1..k_max of its code (tokens [n, K], oriented half-spaces H, o), divided by 1 + 1/m for m
    training images there, so that it estimates the node's variance without bias: [n, k_max], nan where the
    node holds no training image."""
    device, n = X.device, len(Xs)
    k_max = min(k_max, tokens.shape[1])
    out = torch.full((n, k_max), math.nan, device=device)
    bits = torch.zeros(n, dtype=torch.int64)
    uniq = torch.zeros(1, dtype=torch.int64)
    t = torch.arange(len(X))                         # the training images in some held-out image's node,
    node = torch.zeros(len(X), dtype=torch.int64)    # and that node's index in uniq
    step = max(1, ELEMENTS // D)
    for k in range(k_max):
        inv = torch.searchsorted(uniq, bits)
        first = torch.full((len(uniq),), n, dtype=torch.int64).scatter_reduce(0, inv, torch.arange(n), "amin")
        a = tokens[first, k].float().to(device)
        first = first.to(device)
        qn, qo = H[first, k] * a[:, None], o[first, k] * a                         # each node's slice, toward +
        side = torch.cat([((X[t[i:i + step].to(device)] * qn[node[i:i + step].to(device)]).sum(-1)
                           >= qo[node[i:i + step].to(device)]).cpu() for i in range(0, len(t), step)])
        child = uniq[node] | (side.to(torch.int64) << k)
        bits = bits | ((tokens[:, k] > 0).to(torch.int64) << k)
        uniq = torch.unique(bits)
        pos = torch.searchsorted(uniq, child).clamp_max(len(uniq) - 1)
        keep = uniq[pos] == child
        t, node = t[keep], pos[keep]
        count = torch.bincount(node, minlength=len(uniq)).float().to(device)
        nd = node.to(device)
        mean = torch.zeros(len(uniq), D, device=device).index_add_(0, nd, X[t.to(device)]) / count.clamp_min(1)[:, None]
        inv = torch.searchsorted(uniq, bits).to(device)
        c = count[inv]
        e = (Xs - mean[inv]).square().mean(1) / (1 + 1 / c.clamp_min(1))
        out[:, k] = torch.where(c > 0, e, math.nan)
    return out


def rho_mean(values, rho):
    """The rho^k weighted mean over depths k = 1.. of per-depth values [K]."""
    w = rho ** torch.arange(1, len(values) + 1, dtype=values.dtype, device=values.device)
    return ((w * values).sum() / w.sum()).item()


def token_bits(logp):
    """The tokens' bits per pixel, from their log probabilities [n, K]."""
    return (-logp.sum(1) / (D * math.log(2))).mean().item()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--K", type=int, default=32, help="code length (binary tokens)")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--rho", type=float, default=0.95)
    p.add_argument("--passes", type=int, default=4, help="DAgger passes; the first follows the exact targets")
    p.add_argument("--samples", type=int, default=1024, help="training images whose paths each walk follows")
    p.add_argument("--decoder-walks", type=int, default=4, help="walks of the final tree that label the decoder's queries")
    p.add_argument("--val-samples", type=int, default=1000, help="held-out images early stopping watches")
    p.add_argument("--enc-val-depth", type=int, default=16, help="deepest depth of the encoder's early-stopping score")
    p.add_argument("--val-walkers", type=int, default=128,
                   help="held-out images walked once per pass to label queries (diagnostics, and gamma)")
    p.add_argument("--judge-samples", type=int, default=1000)
    p.add_argument("--keep-buffers", type=int, default=2, help="passes' labels to train on (DAgger aggregation)")
    p.add_argument("--epochs", type=int, default=30, help="most epochs per stage (early stopping usually ends sooner)")
    p.add_argument("--patience", type=int, default=2, help="stop after this many epochs without a better held-out score")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--starts", type=lambda s: [int(x) for x in s.split(",")], default=[8, 4, 2])
    p.add_argument("--start-depths", type=lambda s: [int(x) for x in s.split(",")], default=[6, 12])
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--comp-budget", type=int, default=65536, help="density components held per depth")
    p.add_argument("--min-comps", type=int, default=16)
    p.add_argument("--comp-floats", type=int, default=2**28,
                   help="floats of cut state a depth's components may hold (caps the components of deep nodes)")
    p.add_argument("--split-weight", choices=("tail", "uniform"), default="tail",
                   help="the encoder's weight of a split at depth k: its share of the log decompression loss at every "
                        "depth below it, weighted by rho^j (tail), or 1 at every depth (uniform: the leaf's alone)")
    p.add_argument("--gamma", type=float, default=1e-3, help="starting gamma, refit after every pass")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--landmarks", type=int, default=256, help="k-means landmarks the networks read nodes against")
    p.add_argument("--parzen-k", type=int, default=32)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="checkpoints/slicetree.pt")
    p.add_argument("--labels", default="checkpoints/slicetree_labels",
                   help="directory for each walk's labels; pass 0's are reused when its settings match")
    p.add_argument("--log", default="outputs/slicetree.log")
    p.add_argument("--init", default=None,
                   help="a checkpoint to start the encoder and prior from (with --passes 0: train only a decoder on its tree)")
    args = p.parse_args()

    device = get_device(args.device)
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    logf = open(args.log, "a")

    def log(msg):
        msg = f"{time.strftime('%H:%M:%S')} {msg}"
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"=== {time.strftime('%Y-%m-%d')} {vars(args)}")
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    X = dequantize(load_mnist("train", args.data_root)[:args.limit_train], gen)
    Xv = dequantize(load_mnist("val", args.data_root)[:args.limit_val], gen)
    center = X.mean(0)
    X, Xv = (X - center).to(device), (Xv - center).to(device)
    N = len(X)

    t0 = time.time()
    probe = Xv[torch.randperm(len(Xv), generator=gen)[:1000].to(device)]
    bank, _, _ = fit_parzen(X, probe, k=args.parzen_k, r=args.rank, log=log)
    rU, rdiag = root_gaussian(X, args.rank)
    root = Root(center.to(device), rU, rdiag)
    order = torch.randperm(len(Xv), generator=gen).to(device)  # disjoint held-out sets: early stopping, judge, diagnostics
    lik_idx = order[:args.val_samples]
    judge_idx = order[args.val_samples:args.val_samples + args.judge_samples]
    diag_idx = order[args.val_samples + args.judge_samples:][:args.val_walkers]
    Xj = Xv[judge_idx]
    p0_bits = bits_per_dim(mixture_logpdf(Xj, bank, nearest(Xj, X, 64))).mean().item()
    empty_cache(device)
    log(f"round 0 density: {p0_bits:.4f} held-out bits/dim ({time.time() - t0:.0f}s{memory(device)})")

    eig_top = torch.linalg.eigvalsh((X.T @ X / N).cpu())[-1].sqrt().item()
    t0 = time.time()
    landmarks = fit_landmarks(X, args.landmarks, args.rank, bank.kernel_diag, torch.Generator().manual_seed(args.seed + 2))
    models = {}
    for name, outputs in (("encoder", ("mean",)), ("prior", ("share", "projection")), ("decoder", ("mean", "cov"))):
        models[name] = SliceModel(args.rank, args.width, args.layers, args.heads, offset_scale=eig_top,
                                  landmarks=args.landmarks, outputs=outputs).to(device)
        models[name].set_landmarks(*landmarks)
    encoder, prior, decoder = models["encoder"], models["prior"], models["decoder"]
    gamma = args.gamma
    if args.init:
        init = torch.load(args.init, map_location=device, weights_only=False)
        for name in ("encoder", "prior"):
            models[name].load_state_dict(init[name])
        gamma = init["gamma"]
        log(f"encoder and prior from {args.init} (gamma {gamma:.2e})")
    log("networks: " + ", ".join(f"{name} {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M parameters"
                                 for name, m in models.items()) + f"; {args.landmarks} landmarks ({time.time() - t0:.0f}s)")

    def optimizer(model):
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, fused=device.type in ("mps", "cuda"))
    rho_k = lambda k: args.rho**k  # noqa: E731
    if args.split_weight == "tail":  # a split counts at every depth below it
        tail = lambda k: sum(args.rho**j for j in range(k + 1, args.K + 1))  # noqa: E731
    else:  # every split counts once, in the leaf's log decompression loss
        tail = lambda k: 1.0  # noqa: E731
    pass_nets = {"encoder": Net(encoder, optimizer(encoder), "enc", encoder_loss, tail),
                 "prior": Net(prior, optimizer(prior), "prior", prior_loss, rho_k)}
    Xlik, Wval = Xv[lik_idx], Xv[diag_idx]
    stores, history = [], []
    decoder_ready = False

    def walk_held_out(Xs):
        return walk_model(encoder, prior, args.K, args.starts, args.start_depths, args.iters, X=Xs)

    def save(path=None):
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save(dict(encoder=encoder.state_dict(), prior=prior.state_dict(),
                        decoder=decoder.state_dict() if decoder_ready else None,
                        configs={name: m.config for name, m in models.items()}, root=vars(root), args=vars(args),
                        gamma=gamma, history=history, p0_bits=p0_bits), path or args.out)

    prior_val: list = []  # labelled held-out queries on the tree the pass's labels come from

    def pass_score():
        tokens, logp, H, o = walk_held_out(Xlik)
        e = train_estimate(X, Xlik, tokens, H, o, args.enc_val_depth)
        enc = rho_mean(torch.nanmean(e.log(), 0), args.rho)
        objective = held_out_metrics({"prior": pass_nets["prior"]}, prior_val, args, device)["prior"].total("objective")
        cover = (~e[:, -1].isnan()).float().mean().item()
        return ({"encoder": enc, "prior": objective},
                f"encoder {enc:.4f} (log decompression loss, depths 1-{e.shape[1]}; at depth 8 {torch.nanmean(e[:, min(7, e.shape[1] - 1)]).item():.4f}, "
                f"at {e.shape[1]} {torch.nanmean(e[:, -1]).item():.4f} with {cover:.0%} of images covered), prior {objective:.4f} "
                f"(held-out objective; {token_bits(logp):.4f} token bits/dim)")

    def report(tag, t0):
        tokens, logp, H, o = walk_held_out(Xj)
        e = train_estimate(X, Xj, tokens, H, o, args.K)
        depths = [k for k in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128) if k <= args.K]
        line = (f"{tag}: {token_bits(logp):.4f} token bits/dim; decompression loss against the training images in the node "
                + ", ".join(f"{k}: {torch.nanmean(e[:, k - 1]).item():.4f} ({(~e[:, k - 1].isnan()).float().mean().item():.0%})" for k in depths)
                + "; shallow loss " + ", ".join(f"{k + 1}: {v:.4f} ({1 - u:.0%})" for k, (v, u) in enumerate(shallow_loss(tokens, Xj))))
        entry = dict(tag=tag, token_bits=token_bits(logp), train_estimate=torch.nanmean(e, 0).tolist())
        if decoder_ready:
            means, _, lp = decode(decoder, root, tokens, H, o, X=Xj)
            assert lp is not None
            bits = bits_per_dim(logp.sum(1) + lp[:, -1]).mean().item()
            err = (means - Xj[:, None]).square().mean(-1).mean(0)
            line += (f"; {bits:.4f} held-out bits/dim (round 0 density {p0_bits:.4f}); decoder's mean-image error by depth "
                     + ", ".join(f"{k}: {err[k].item():.4f}" for k in depths))
            entry.update(bits=bits, err=err.tolist())
        log(line + f" ({time.time() - t0:.0f}s)")
        history.append(entry)

    for pas in range(args.passes):
        t0 = time.time()
        samp = torch.randperm(N, generator=gen)[:args.samples].to(device)
        directory = os.path.join(args.labels, f"pass{pas}")
        key = {"version": 5, **{k: getattr(args, k) for k in ("K", "rank", "rho", "samples", "starts", "start_depths",
                                                              "iters", "comp_budget", "min_comps", "gamma",
                                                              "parzen_k", "limit_train", "limit_val", "seed", "landmarks")}}
        store = LabelStore.cached(directory, key) if pas == 0 else None
        if store is not None:
            log(f"pass 0: labels from {directory}")
        else:
            store = LabelStore.fresh(directory)
            with torch.no_grad():
                walk(args, bank, root, X, X[samp], (encoder, prior), pas == 0, gamma, gen, log, store, args.comp_budget,
                     roles=("enc", "prior", "dec") if args.decoder_walks else ("enc", "prior"))
            store.finish(key if pas == 0 else None)
        log(f"pass {pas} ({'exact' if pas == 0 else 'model'} Lloyd): {store.size()} labelled queries ({time.time() - t0:.0f}s)")
        stores = (stores + [store])[-args.keep_buffers:]
        if pas == 0:  # the prior's held-out queries: on the exact tree, as its first labels are
            t0 = time.time()
            prior_val[:] = held_out_labels(pass_nets, args, bank, root, X, Wval, gamma, log, teacher=True, roles=("prior",))[0]
            log(f"  held-out queries on the exact tree, for the prior's early stopping ({time.time() - t0:.0f}s)")
        fit(pass_nets, stores, args, device, pass_score, log, save)
        t0 = time.time()
        labelled, records = held_out_labels(pass_nets, args, bank, root, X, Wval, gamma, log)
        diagnostics = held_out_metrics(pass_nets, labelled, args, device)
        prior_val[:] = labelled  # the next pass's labels come from this tree
        log("  labelled held-out queries: " + "; ".join(f"{name}: {m.text()}" for name, m in diagnostics.items())
            + f" ({time.time() - t0:.0f}s; depth quarters in brackets)")
        fitted = fit_gamma(records, N)
        log(f"  gamma {gamma:.2e} -> {fitted if fitted is None else f'{fitted:.2e}'}")
        gamma = fitted or gamma
        report(f"pass {pas}", time.time())
        save()
        save(args.out.replace(".pt", f"_pass{pas}.pt"))

    # the decoder: on the final tree, labelled along many more paths
    if not args.decoder_walks:  # no Gaussian decoder (a categorical one is trained by train_slicetree_levels.py)
        return
    dec_stores = []
    for i in range(args.decoder_walks):
        t0 = time.time()
        samp = torch.randperm(N, generator=gen)[:args.samples].to(device)
        store = LabelStore.fresh(os.path.join(args.labels, f"decoder{i}"))
        with torch.no_grad():
            walk(args, bank, root, X, X[samp], (encoder, prior), False, gamma, gen, log, store, args.comp_budget, roles=("dec",))
        store.finish()
        dec_stores.append(store)
        log(f"decoder walk {i}: {store.size()} labelled queries ({time.time() - t0:.0f}s)")
    tokens, logp, H, o = walk_held_out(Xlik)

    def decoder_score():
        _, _, lp = decode(decoder, root, tokens, H, o, X=Xlik)
        assert lp is not None
        per_depth = bits_per_dim(lp[:, 1:]).mean(0)
        score = rho_mean(per_depth, args.rho)
        return {"decoder": score}, (f"decoder {score:.4f} bits/dim (node Gaussians, depths weighted by rho^k); "
                                    f"leaf {per_depth[-1].item():.4f}, with the tokens {bits_per_dim(logp.sum(1) + lp[:, -1]).mean().item():.4f}")
    decoder_ready = True
    fit({"decoder": Net(decoder, optimizer(decoder), "dec", decoder_loss, rho_k)}, dec_stores, args, device,
        decoder_score, log, save)
    report("final", time.time())
    save()


if __name__ == "__main__":
    main()
