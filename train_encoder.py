"""Stage 1: build the staged encoder (dvae/staged.py) one code position at a time.

For each position k (prefix length t = k-1):
  1. Partitions. Sample P anchor images, each with one out-of-fold code (from the positions built so far,
     on images they weren't trained on). Every anchor gets R reference draws, split evenly over its t+1
     groups by the length of the prefix a code shares with the anchor's: each image is drawn in proportion
     to its exact out-of-fold probability of landing in the group, and each draw weighs the group's mass
     over its draws. For every anchor, build a partition of its references with Hartigan's method on the
     partition's implied marginal decompression loss, with each reference's code truncated against the
     anchor's (dvae/partition.py). Each partition labels any (image, code) by the class whose loss rises
     least if the image joins it.
  2. Encoders. Train num_folds + 1 StageNets with one head per partition, by cross-entropy against the
     partitions' labels: fold model f on the training images outside fold f, the final model on all of
     them. Each minibatch draws one fresh out-of-fold code per image (memoized). All copies share the
     epoch count, chosen by the pooled held-out cross-entropy.
  3. Costs. For every training image, with its held-out encoder, its term of the marginal decompression
     loss under each partition (exact, all training images as references).
  4. Selectors. Train num_folds + 1 SelectorNets to predict those costs from the code alone (MSE), with
     the same fold split and shared epochs, early-stopped on the held-out marginal decompression loss of
     the fold copies' choices (the cost of the partition each picks, pooled over images). Afterwards every
     copy uses the final selector, so each token means the same thing in every copy. The encoder uses the
     partition with the lowest predicted cost.
  5. Report the marginal decompression loss on the val split: per the selector's choices, for the single
     best partition, and with no code.

Saves the encoder after every position (--out, usable by the decoder/prior/QA scripts) and the full build
state (--state, every fold model) so --resume can continue with the next position.
"""

import argparse
import copy
import math
import time
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F

from dvae.common import add_common_args, get_device, save_checkpoint
from dvae.data import load_mnist, to_float
from dvae.partition import AnchorPartition, node_costs
from dvae.staged import SelectorNet, StagedEncoder, StageNet, node_ids

FORWARD_CHUNK = 2048  # images per forward pass when scoring or sampling


def log(msg):
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@contextmanager
def timed(what):
    """Log the start of an expensive step and how long it took."""
    log(f"{what} ...")
    t0 = time.time()
    yield
    log(f"{what}: done in {time.time() - t0:.1f}s")


class Ticker:
    """Throttled progress lines inside a long loop: at most one per interval seconds."""

    def __init__(self, interval):
        self.interval, self.last = interval, time.time()

    def __call__(self, msg):
        if time.time() - self.last >= self.interval:
            self.last = time.time()
            log(msg)


def forward_chunked(model, x, prefix):
    """model(x, prefix) in chunks -> logits [N, n_heads, V]."""
    return torch.cat([model(x[i:i + FORWARD_CHUNK], prefix[i:i + FORWARD_CHUNK]) for i in range(0, len(x), FORWARD_CHUNK)])


class CodeSampler:
    """Samples codes through the positions built so far.

    Training images go through the out-of-fold encoder and selector of their own fold, and the next-token
    distribution of every (image, prefix) visited is memoized: positions are frozen once built and the
    selector's choice depends on the prefix alone, so it never changes. The memo holds at most memo_cap
    entries; when full it evicts the prefixes least likely to be reached. Other images (val) go through the
    final models, without the memo. Each position's image features for every training image are cached
    the first time they're needed, for the same reason.
    """

    def __init__(self, images, fold_of, vocab_size, n_folds, memo_cap):
        self.images, self.fold_of, self.V, self.n_folds = images, fold_of, vocab_size, n_folds
        self.memo_cap, self.memo, self.positions = memo_cap, {}, []
        self.features: list[torch.Tensor | None] = []

    def add_position(self, encoders, selectors):
        for m in list(encoders) + list(selectors):
            m.eval().requires_grad_(False)
        self.positions.append((encoders, selectors))
        self.features.append(None)

    @torch.no_grad()
    def train_features(self, t):
        """Position t's image features [N, width] of every training image, from its out-of-fold encoder."""
        cached = self.features[t]
        if cached is None:
            encoders, _ = self.positions[t]
            device = self.images.device
            cached = torch.empty(len(self.images), encoders[0].head[0].in_features, device=device)
            for m in range(self.n_folds):
                rows = (self.fold_of == m).nonzero().squeeze(1).to(device)
                for i in range(0, len(rows), FORWARD_CHUNK):
                    r = rows[i:i + FORWARD_CHUNK]
                    cached[r] = encoders[m].features(self.images[r])
            self.features[t] = cached
        return cached

    def _probs(self, t, x, prefix, model_idx):
        """Next-token distributions [n, V] (on CPU) from position t's model_idx[i] encoder and selector."""
        encoders, selectors = self.positions[t]
        out = torch.empty(len(x), self.V)
        for m in model_idx.unique().tolist():
            rows = (model_idx == m).nonzero().squeeze(1)
            rows_dev = rows.to(x.device)
            logits = forward_chunked(encoders[m], x[rows_dev], prefix[rows_dev])
            head = selectors[m].choose(prefix[rows_dev])
            out[rows] = logits[torch.arange(len(rows_dev), device=x.device), head].softmax(-1).cpu()
        return out

    def _next_probs(self, t, x, prefix, model_idx, keys, path):
        probs = torch.empty(len(x), self.V)
        hits, hit_vals, misses = [], [], []
        for row, key in enumerate(keys):
            value = self.memo.get(key)
            if value is None:
                misses.append(row)
            else:
                hits.append(row)
                hit_vals.append(value[0])
        if hits:
            probs[hits] = torch.tensor(hit_vals)
        if misses:
            rows = torch.tensor(misses)
            rows_dev = rows.to(x.device)
            p = self._probs(t, x[rows_dev], prefix[rows_dev], model_idx[rows])
            probs[rows] = p
            for j, row in enumerate(misses):
                self.memo[keys[row]] = (tuple(p[j].tolist()), float(path[row]))
        return probs

    @torch.no_grad()
    def sample_train(self, idx, depth, generator):
        """One code [B, depth] per training image idx, drawn from its out-of-fold distribution."""
        B, device = len(idx), idx.device
        prefix = torch.zeros(B, 0, dtype=torch.long, device=device)
        ids, folds = idx.tolist(), self.fold_of[idx.cpu()]
        node = torch.zeros(B, dtype=torch.long)
        path = torch.ones(B, dtype=torch.float64)
        for t in range(depth):
            keys = list(zip(ids, [t] * B, node.tolist()))
            probs = self._next_probs(t, self.images[idx], prefix, folds, keys, path)
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            path = path * probs[torch.arange(B), nxt].double()
            node = node * self.V + nxt
            prefix = torch.cat([prefix, nxt.to(device)[:, None]], 1)
        if len(self.memo) > self.memo_cap:  # keep the prefixes most likely to be reached
            with timed(f"memo full ({len(self.memo)} entries): evicting the least likely prefixes"):
                keep = sorted(self.memo.items(), key=lambda item: item[1][1], reverse=True)[: int(0.9 * self.memo_cap)]
                self.memo = dict(keep)
        return prefix

    def sample_all_train(self, depth, generator, progress=None):
        """One out-of-fold code for every training image -> [N, depth]."""
        n = len(self.images)
        idx = torch.arange(n, device=self.images.device)
        out = []
        for start in range(0, n, FORWARD_CHUNK):
            if progress is not None:
                progress(f"  codes for {start}/{n} images (memo {len(self.memo)})")
            out.append(self.sample_train(idx[start:start + FORWARD_CHUNK], depth, generator))
        return torch.cat(out)

    @torch.no_grad()
    def stage_probs(self, t, prefixes):
        """Position t's next-token distributions [n, N, V] of every training image (out-of-fold models) after
        each of the prefixes [n, t], from the cached image features."""
        encoders, selectors = self.positions[t]
        device = self.images.device
        features = self.train_features(t)
        out = torch.empty(len(prefixes), len(self.images), self.V, device=device)
        for m in range(self.n_folds):
            rows = (self.fold_of == m).nonzero().squeeze(1).to(device)
            heads = selectors[m].choose(prefixes).tolist()
            for i in range(0, len(rows), FORWARD_CHUNK):
                r = rows[i:i + FORWARD_CHUNK]
                feats = features[r]
                for j, head in enumerate(heads):
                    out[j, r] = encoders[m].logits(feats, prefixes[j:j + 1])[:, head].softmax(-1)
        return out

    @torch.no_grad()
    def sample_final(self, x, depth, generator):
        """Codes [N, depth] for other images, through the final models."""
        prefix = torch.zeros(len(x), 0, dtype=torch.long, device=x.device)
        final = torch.full((len(x),), self.n_folds, dtype=torch.long)
        for t in range(depth):
            nxt = torch.multinomial(self._probs(t, x, prefix, final), 1, generator=generator).squeeze(1)
            prefix = torch.cat([prefix, nxt.to(x.device)[:, None]], 1)
        return prefix


@torch.no_grad()
def draw_references(sampler, anchor_codes, R, generator, progress):
    """Reference draws for every anchor, split evenly over its t+1 truncated groups by shared prefix length.

    Group s < t holds the codes that share exactly s tokens with the anchor; group t holds the full match.
    An image's mass in group s is its exact out-of-fold probability of landing there,
    q(a[:s] | y) * (1 - q(a_s | y, a[:s])), so each group gets R/(t+1) draws however small its mass. Images
    are drawn with replacement in proportion to their mass, a mismatch group's token at position s among the
    non-anchor tokens in proportion to q, and every draw weighs (group mass) / (draws in the group).
    Returns, per anchor, (image indices, codes [n, t], weights), and the group masses [P, t+1].
    """
    P, t = anchor_codes.shape
    device, N = sampler.images.device, len(sampler.images)
    reach = torch.ones(P, N, device=device)  # q(a[:u] | y): the probability of following the anchor so far
    masses = torch.zeros(P, t + 1)
    refs: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = [[] for _ in range(P)]
    for u in range(t + 1):
        n_draws = R // (t + 1) + (u < R % (t + 1))
        if u < t:
            progress(f"  group probabilities at prefix length {u}/{t}")
            prefixes = anchor_codes[:, :u].cpu()
            if u == 0:
                uniq, inv = prefixes[:1], torch.zeros(P, dtype=torch.long)
            else:
                uniq, inv = torch.unique(prefixes, dim=0, return_inverse=True)
            probs = sampler.stage_probs(u, uniq.to(device))  # [n_unique, N, V]
        for a in range(P):
            if u < t:
                other = probs[inv[a]].clone()
                other[:, anchor_codes[a, u]] = 0.0  # leaving the anchor's path at position u
                mass = reach[a] * other.sum(1)
            else:
                other, mass = None, reach[a]
            masses[a, u] = mass.sum().item()
            if masses[a, u] > 0:
                idx = torch.multinomial(mass.cpu(), n_draws, replacement=True, generator=generator).to(device)
                codes = anchor_codes[a].expand(n_draws, t).clone()
                if other is not None:
                    codes[:, u] = torch.multinomial(other[idx].cpu(), 1, generator=generator).squeeze(1).to(device)
                refs[a].append((idx, codes, torch.full((n_draws,), masses[a, u].item() / n_draws, device=device)))
            if u < t:
                reach[a] *= probs[inv[a], :, anchor_codes[a, u]]
    out = [tuple(torch.cat(parts) for parts in zip(*r)) for r in refs]
    return out, masses


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--vocab-size", type=int, default=2)
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--folds", type=int, default=8)
    p.add_argument("--anchors", type=int, default=32, help="P: anchor images, one partition each")
    p.add_argument("--references", type=int, default=8192,
                   help="R: reference draws per anchor, split evenly over its truncated groups")
    p.add_argument("--hartigan-rounds", type=int, default=200, help="cap on batched Hartigan move rounds per partition")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=20, help="max encoder epochs per position")
    p.add_argument("--patience", type=int, default=1)
    p.add_argument("--selector-width", type=int, default=128)
    p.add_argument("--selector-batch-size", type=int, default=1024)
    p.add_argument("--selector-lr", type=float, default=1e-3)
    p.add_argument("--selector-epochs", type=int, default=100, help="max selector epochs per position")
    p.add_argument("--selector-patience", type=int, default=5)
    p.add_argument("--memo-cap", type=int, default=2_000_000)
    p.add_argument("--log-interval", type=float, default=5.0, help="seconds between progress lines")
    p.add_argument("--limit-train", type=int, default=None, help="use only the first N training images (smoke tests)")
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--out", default="checkpoints/encoder.pt")
    p.add_argument("--state", default="checkpoints/encoder_build.pt")
    p.add_argument("--resume", action="store_true", help="continue from --state with the next position")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    with timed("loading MNIST"):
        X = to_float(load_mnist("train", args.data_root, device)[: args.limit_train])
        Xv = to_float(load_mnist("val", args.data_root, device)[: args.limit_val])
    mean = X.flatten(1).mean(0)
    Xc, Xvc = X.flatten(1) - mean, Xv.flatten(1) - mean  # centered pixels, for the objective's precision
    N, Nv, V, Fo, P = len(X), len(Xv), args.vocab_size, args.folds, args.anchors
    R = args.references
    log(f"device {device}, {N} training images in {Fo} folds, {Nv} val images, V={V}, k_max={args.k_max}, "
        f"{P} anchors, {R} reference draws per anchor")

    state: dict[str, Any] = {"args": vars(args), "stages": [], "history": []}
    if args.resume:
        state = torch.load(args.state, map_location="cpu", weights_only=False)
        fold_of = state["fold_of"]
        log(f"resuming from {args.state}: {len(state['stages'])} positions built")
    else:
        fold_of = torch.randperm(N, generator=torch.Generator().manual_seed(args.seed)) % Fo
        state["fold_of"] = fold_of
    sampler = CodeSampler(X, fold_of, V, Fo, args.memo_cap)
    cfgs, final_encoders, final_selectors = [], [], []
    if state["stages"]:
        log(f"rebuilding the {len(state['stages'])} saved positions ({Fo + 1} encoders and selectors each)")
    for t, st in enumerate(state["stages"]):
        cfg = st["cfg"]
        encs, sels = [], []
        for enc_state, sel_state in zip(st["encoder_states"], st["selector_states"]):
            e = StageNet(V, t, cfg["n_heads"], cfg["width"]).to(device)
            e.load_state_dict(enc_state)
            s = SelectorNet(V, t, cfg["n_heads"], cfg["selector_width"]).to(device)
            s.load_state_dict(sel_state)
            encs.append(e)
            sels.append(s)
        sampler.add_position(encs, sels)
        cfgs.append(cfg)
        final_encoders.append(st["encoder_states"][Fo])
        final_selectors.append(st["selector_states"][Fo])

    # mean distance with no code, for reference: val image vs all training images
    k0 = ((Xvc.pow(2).sum(1) - 2 * Xvc @ Xc.mean(0) + Xc.pow(2).sum(1).mean()) / Xc.shape[1]).mean().item()
    all_idx = torch.arange(N, device=device)
    train_sets = [all_idx[(fold_of != f).to(device)] for f in range(Fo)] + [all_idx]
    held_out = [(fold_of == f).nonzero().squeeze(1).to(device) for f in range(Fo)]

    try:
        for t in range(len(state["stages"]), args.k_max):
            t0 = time.time()
            log(f"=== position {t + 1} of {args.k_max} (prefix length {t})")
            gen = torch.Generator().manual_seed(args.seed * 1000 + t)
            tick = Ticker(args.log_interval)

            # 1. anchor partitions
            with timed(f"sampling {P} anchors with out-of-fold codes"):
                anchor_idx = torch.randperm(N, generator=gen)[:P].to(device)
                anchor_codes = sampler.sample_train(anchor_idx, t, gen)
            with timed(f"drawing {R} references per anchor, evenly over its {t + 1} groups by shared prefix length"):
                refs, masses = draw_references(sampler, anchor_codes, R, gen, tick)
            log("group masses in images (mean over anchors, shared prefix length 0.." + f"{t}, last = full match): "
                + " ".join(f"{m:.1f}" for m in masses.mean(0).tolist()))
            with timed(f"building {P} anchor partitions (Hartigan on {R} references each)"):
                parts = []
                for a, (ref_idx, ref_codes, ref_w) in enumerate(refs):
                    tick(f"  partition {a + 1}/{P}")
                    parts.append(AnchorPartition(anchor_codes[a], Xc[ref_idx], ref_codes, V, gen, args.hartigan_rounds,
                                                 ref_w))
            losses = [pt.total_loss() / masses[a].sum().item() for a, pt in enumerate(parts)]
            log(f"partitions: {sum(len(pt.table.nodes) for pt in parts) / P:.1f} groups on average, "
                f"{sum(pt.rounds for pt in parts) / P:.1f} Hartigan rounds on average, estimated loss per image "
                f"{min(losses):.5f}..{max(losses):.5f}")

            def targets(idx, codes):
                return torch.stack([pt.labels(Xc[idx], codes) for pt in parts], 1)

            with timed("held-out codes and partition labels for every training image"):
                eval_codes = sampler.sample_all_train(t, torch.Generator().manual_seed(args.seed * 1000 + t + 500), tick)
                eval_targets = targets(all_idx, eval_codes)

            # 2. encoders: F+1 multi-head StageNets, lockstep epochs, early stopping on held-out cross-entropy
            encoders = [StageNet(V, t, P, args.width).to(device) for _ in range(Fo + 1)]
            opts = [torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.weight_decay) for m in encoders]
            best: dict[str, Any] = {"score": math.inf}
            bad = 0
            for epoch in range(1, args.epochs + 1):
                with timed(f"encoder epoch {epoch}: training {Fo + 1} models with {P} heads each"):
                    losses = []
                    for m, (model, opt) in enumerate(zip(encoders, opts)):
                        model.train()
                        order = train_sets[m][torch.randperm(len(train_sets[m]), generator=gen).to(device)]
                        for b in order.split(args.batch_size):
                            codes = sampler.sample_train(b, t, gen)
                            loss = F.cross_entropy(model(X[b], codes).reshape(-1, V), targets(b, codes).reshape(-1))
                            opt.zero_grad(set_to_none=True)
                            loss.backward()
                            opt.step()
                            losses.append(loss.item())
                            tick(f"  model {m + 1}/{Fo + 1} cross-entropy {sum(losses[-50:]) / len(losses[-50:]):.4f} "
                                 f"(memo {len(sampler.memo)})")
                with torch.no_grad():
                    total = 0.0
                    for f in range(Fo):
                        encoders[f].eval()
                        logits = forward_chunked(encoders[f], X[held_out[f]], eval_codes[held_out[f]])
                        total += F.cross_entropy(logits.reshape(-1, V), eval_targets[held_out[f]].reshape(-1),
                                                 reduction="sum").item()
                    held_ce = total / (N * P)
                improved = held_ce < best["score"]
                log(f"encoder epoch {epoch}: train cross-entropy {sum(losses) / len(losses):.4f}, held-out {held_ce:.4f}"
                    f"{'  *best*' if improved else ''}")
                if improved:
                    best = {"score": held_ce, "epoch": epoch, "states": [copy.deepcopy(m.state_dict()) for m in encoders]}
                    bad = 0
                else:
                    bad += 1
                    if bad >= args.patience:
                        break
            for m, s in zip(encoders, best["states"]):
                m.load_state_dict(s)
                m.eval()
            encoder_epochs = best["epoch"]

            # 3. held-out marginal decompression loss of every training image under every partition
            with torch.no_grad(), timed(f"held-out marginal decompression loss of every training image under each of {P} "
                                        "partitions"):
                q = torch.empty(N, P, V, device=device)
                for f in range(Fo):
                    q[held_out[f]] = forward_chunked(encoders[f], X[held_out[f]], eval_codes[held_out[f]]).softmax(-1)
                eval_ids = node_ids(eval_codes, V)
                ones = torch.ones(N, device=device)
                cost = torch.empty(N, P, device=device)
                for a in range(P):
                    tick(f"  partition {a + 1}/{P}")
                    cost[:, a] = node_costs(Xc, all_idx, eval_ids, q[:, a], Xc, all_idx, eval_ids, q[:, a], ones, all_idx)
            mean_cost = cost.mean(0)
            log(f"held-out loss per partition: best {mean_cost.min():.5f}, worst {mean_cost.max():.5f}, "
                f"per-image best (oracle) {cost.min(1).values.mean():.5f}")

            # 4. selectors: F+1 SelectorNets regressing the costs from the code, early stopping on the held-out
            # loss of their choices
            selectors = [SelectorNet(V, t, P, args.selector_width).to(device) for _ in range(Fo + 1)]
            for s in selectors:
                with torch.no_grad():
                    s.out.bias.copy_(mean_cost)  # start from each partition's mean cost
            sel_opts = [torch.optim.AdamW(s.parameters(), lr=args.selector_lr) for s in selectors]
            best_sel: dict[str, Any] = {"score": math.inf}
            bad = 0
            with timed(f"training {Fo + 1} selectors"):
                for epoch in range(1, args.selector_epochs + 1):
                    for m, (sel, opt) in enumerate(zip(selectors, sel_opts)):
                        sel.train()
                        order = train_sets[m][torch.randperm(len(train_sets[m]), generator=gen).to(device)]
                        for b in order.split(args.selector_batch_size):
                            loss = F.mse_loss(sel(eval_codes[b]), cost[b])
                            opt.zero_grad(set_to_none=True)
                            loss.backward()
                            opt.step()
                    with torch.no_grad():
                        mse, choice = 0.0, 0.0
                        for f in range(Fo):
                            pred = selectors[f].eval()(eval_codes[held_out[f]])
                            mse += F.mse_loss(pred, cost[held_out[f]], reduction="sum").item()
                            choice += cost[held_out[f]].gather(1, pred.argmin(-1, keepdim=True)).sum().item()
                        mse, choice = mse / (N * P), choice / N
                    improved = choice < best_sel["score"]
                    tick(f"  selector epoch {epoch}: held-out loss of choices {choice:.4f}, MSE {mse:.3e}"
                         f"{'  *best*' if improved else ''}")
                    if improved:
                        best_sel = {"score": choice, "mse": mse, "epoch": epoch,
                                    "states": [copy.deepcopy(s.state_dict()) for s in selectors]}
                        bad = 0
                    else:
                        bad += 1
                        if bad >= args.selector_patience:
                            break
            for s, st in zip(selectors, best_sel["states"]):
                s.load_state_dict(st)
                s.eval()
            log(f"selectors: best epoch {best_sel['epoch']}, held-out loss of choices {best_sel['score']:.4f}, "
                f"MSE {best_sel['mse']:.3e}")
            # Every copy codes with the final selector from here on: the fold selectors only chose the epoch.
            # One choice per code keeps each token's meaning the same in every copy (partitions can be tied
            # mirror images, so independent choices could flip it).
            for s in selectors[:Fo]:
                s.load_state_dict(selectors[Fo].state_dict())
            sampler.add_position(encoders, selectors)

            # 5. report on the val split: val images coded by the final models, training images (coded out of
            # fold) as references; the selector's choices vs the single best partition vs no code
            with torch.no_grad(), timed("val report (final encoder and selector, val split)"):
                val_codes = sampler.sample_final(Xv, t, torch.Generator().manual_seed(args.seed + 7 + t))
                val_ids = node_ids(val_codes, V)
                qv = forward_chunked(encoders[Fo], Xv, val_codes).softmax(-1)
                vrows = torch.arange(Nv, device=device)
                no_ref = torch.full((Nv,), -1, dtype=torch.long, device=device)
                single = int(mean_cost.argmin())
                report = {}
                for label, pa, pr in [("selector", selectors[Fo].choose(val_codes), selectors[Fo].choose(eval_codes)),
                                      ("single best", torch.full((Nv,), single, device=device),
                                       torch.full((N,), single, device=device))]:
                    report[label] = node_costs(Xvc, vrows, val_ids, qv[vrows, pa], Xc, all_idx, eval_ids,
                                               q[all_idx, pr], ones, no_ref).mean().item()
                chosen = torch.bincount(selectors[Fo].choose(eval_codes), minlength=P)
            entry = {"position": t + 1, "encoder_epochs": encoder_epochs, "selector_epochs": best_sel["epoch"],
                     "held_out_cross_entropy": best["score"], "held_out_loss_per_partition": mean_cost.tolist(),
                     "selector_choice_loss": best_sel["score"],
                     "selector_mse": best_sel["mse"], "val_selector": report["selector"],
                     "val_single_best": report["single best"], "k0": k0, "seconds": time.time() - t0}
            state["history"].append(entry)
            log(f"position {t + 1} done in {entry['seconds']:.0f}s: val loss at k={t + 1}: selector {report['selector']:.5f}, "
                f"single best partition {report['single best']:.5f}, no code {k0:.5f}; partitions the selector uses "
                f"(training codes): {(chosen > 0).sum().item()} of {P}")

            with timed(f"saving {args.out} and {args.state}"):
                cfg = dict(n_heads=P, width=args.width, selector_width=args.selector_width)
                state["stages"].append({"cfg": cfg,
                                        "encoder_states": [copy.deepcopy(m.state_dict()) for m in encoders],
                                        "selector_states": [copy.deepcopy(s.state_dict()) for s in selectors]})
                torch.save(state, args.state)
                cfgs.append(cfg)
                final_encoders.append(copy.deepcopy(encoders[Fo].state_dict()))
                final_selectors.append(copy.deepcopy(selectors[Fo].state_dict()))
                encoder = StagedEncoder(V, len(cfgs), cfgs)
                for net, s in zip(encoder.nets, final_encoders):
                    net.load_state_dict(s)
                for net, s in zip(encoder.selectors, final_selectors):
                    net.load_state_dict(s)
                save_checkpoint(args.out, "encoder", encoder, args=vars(args), history=state["history"])
    except KeyboardInterrupt:
        log(f"interrupted; {len(state['stages'])} completed positions are in {args.state} (continue with --resume)")


if __name__ == "__main__":
    main()
