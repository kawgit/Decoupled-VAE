"""Stage 1: build the staged encoder (dvae/staged.py) one code position at a time.

For each position t = 0 .. k_max-1 (prefix length t):
  1. Draw S = --prefix-samples distinct prefixes per training image without replacement (stochastic beam
     search over the out-of-fold models of earlier stages), each with an importance weight.
  2. Fit --num-candidates candidate partitions (dvae/partition.py): each draws a sharing level m uniformly
     from 0..t and fits a weighted V-means codebook per m-prefix group. Score every candidate by its
     in-sample objective, drop near-duplicates (weighted ARI), keep the best --keep-candidates.
  3. Train num_folds + 1 copies of a StageNet with one head per kept candidate, by cross-entropy against
     the candidates' labels: fold model f on the images outside fold f, the final model on all of them.
     Each batch draws one prefix per image from the out-of-fold models (memoized), so later positions
     learn from the mistakes earlier positions really make on unseen images.
  4. After every epoch, score every head on each fold's held-out images with the decompression objective
     (exact, all training entries as references), pick the best head per prefix, and keep the epoch with
     the best total. The decision is shared by all copies, so the final model trains for as many epochs.
  5. The final model with that per-prefix head selection becomes position t. Its honest score is reported
     on the separate val split, for the per-prefix selection and for the single best head.

Saves the encoder after every position (--out, usable by the decoder/prior/QA scripts) and the full build
state (--state, every fold model) so --resume can continue with the next position.
"""

import argparse
import copy
import math
import time
from collections import Counter
from typing import Any

import torch
import torch.nn.functional as F

from dvae.common import add_common_args, get_device, save_checkpoint
from dvae.data import load_mnist, to_float
from dvae.partition import (Entries, candidate_proxy, costs_by_group, fit_candidate, gumbel, node_costs,
                            weighted_node_ari, weighted_top_direction)
from dvae.staged import Selector, StagedEncoder, StageNet, node_ids

FORWARD_CHUNK = 2048  # images per forward pass when scoring or sampling


def forward_chunked(model, x, prefix):
    """model(x, prefix) in chunks -> logits [N, n_heads, V]."""
    return torch.cat([model(x[i:i + FORWARD_CHUNK], prefix[i:i + FORWARD_CHUNK]) for i in range(0, len(x), FORWARD_CHUNK)])


class PrefixSampler:
    """Samples prefixes through the positions built so far.

    Training images go through the out-of-fold model of their own fold, and the next-token distribution of
    every (image, prefix) visited is memoized: positions are frozen once built, so it never changes. The memo
    holds at most memo_cap entries; when full it evicts the prefixes least likely to be reached. Other images
    (val) go through the final models, without the memo.
    """

    def __init__(self, images, fold_of, vocab_size, n_folds, memo_cap):
        self.images, self.fold_of, self.V, self.n_folds = images, fold_of, vocab_size, n_folds
        self.memo_cap, self.memo, self.stages = memo_cap, {}, []

    def add_stage(self, models, selector):
        for m in models:
            m.eval().requires_grad_(False)
        self.stages.append((models, selector))

    def _probs(self, t, x, prefix, model_idx):
        """Next-token distributions [n, V] (CPU) from position t's model_idx[i] model."""
        models, selector = self.stages[t]
        out = torch.empty(len(x), self.V)
        for m in model_idx.unique().tolist():
            rows = (model_idx == m).nonzero().squeeze(1)
            rows_dev = rows.to(x.device)
            logits = forward_chunked(models[m], x[rows_dev], prefix[rows_dev])
            head = selector(node_ids(prefix[rows_dev], self.V))
            out[rows] = logits[torch.arange(len(rows_dev), device=x.device), head].softmax(-1).cpu()
        return out

    def next_probs(self, t, x, prefix, model_idx, keys=None, path=None):
        """[n, V] next-token distributions; with keys (training images), served from and stored in the memo."""
        if keys is None:
            return self._probs(t, x, prefix, model_idx)
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
                self.memo[keys[row]] = (tuple(p[j].tolist()), float(path[row]) if path is not None else 0.0)
        return probs

    def _evict(self):
        if len(self.memo) > self.memo_cap:  # keep the prefixes most likely to be reached
            keep = sorted(self.memo.items(), key=lambda item: item[1][1], reverse=True)[: int(0.9 * self.memo_cap)]
            self.memo = dict(keep)

    @torch.no_grad()
    def sample_train(self, idx, depth, generator):
        """One prefix [B, depth] per training image idx, drawn from its out-of-fold distribution."""
        B, device = len(idx), idx.device
        prefix = torch.zeros(B, 0, dtype=torch.long, device=device)
        ids, folds = idx.tolist(), self.fold_of[idx.cpu()]
        node = torch.zeros(B, dtype=torch.long)
        path = torch.ones(B, dtype=torch.float64)
        for t in range(depth):
            keys = list(zip(ids, [t] * B, node.tolist()))
            probs = self.next_probs(t, self.images[idx], prefix, folds, keys, path)
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            path = path * probs[torch.arange(B), nxt].double()
            node = node * self.V + nxt
            prefix = torch.cat([prefix, nxt.to(device)[:, None]], 1)
        self._evict()
        return prefix

    @torch.no_grad()
    def sample_distinct(self, x, depth, S, generator, idx=None):
        """S distinct prefixes per image, sampled without replacement (stochastic beam search).

        x: images [B, 1, 28, 28]; idx: their training indices (out-of-fold models, memoized), or None for
        the final models. Returns prefixes [B, W, depth] and importance weights [B, W] with W = min(S, V**depth):
        weight = q(z) / pi(z), pi(z) = 1 - exp(-q(z) e^-kappa), kappa the (S+1)-th largest perturbed log-prob,
        so weighted sums over the samples are unbiased for sums over the prefix distribution (exact, with
        weight q(z), when every prefix fits in the sample).
        """
        B, V, device = len(x), self.V, x.device
        model_idx = torch.full((B,), self.n_folds, dtype=torch.long) if idx is None else self.fold_of[idx.cpu()]
        img_ids = None if idx is None else idx.tolist()
        beam_prefix = torch.zeros(B, 1, 0, dtype=torch.long, device=device)
        beam_node = torch.zeros(B, 1, dtype=torch.long)
        phi = torch.zeros(B, 1, dtype=torch.float64)  # log q(prefix)
        G = gumbel((B, 1), generator)                 # perturbed log q(prefix)
        for t in range(depth):
            n = beam_node.shape[1]
            keys = None if img_ids is None else \
                list(zip([i for i in img_ids for _ in range(n)], [t] * (B * n), beam_node.flatten().tolist()))
            probs = self.next_probs(t, x.repeat_interleave(n, 0), beam_prefix.reshape(B * n, t),
                                    model_idx.repeat_interleave(n), keys, phi.exp().flatten())
            phi_child = phi[..., None] + probs.view(B, n, V).double().clamp_min(1e-30).log()
            g = phi_child + gumbel((B, n, V), generator)
            Z = g.max(-1, keepdim=True).values
            v = G[..., None] - g + torch.log1p(-torch.exp(g - Z))  # children's Gumbels conditioned on max = G
            G_child = G[..., None] - v.clamp_min(0) - torch.log1p(torch.exp(-v.abs()))
            top = G_child.flatten(1).topk(min(S + 1, n * V), dim=1)
            parent, token = top.indices // V, top.indices % V
            beam_node = beam_node.gather(1, parent) * V + token
            kept = beam_prefix.gather(1, parent.to(device)[..., None].expand(-1, -1, t))
            beam_prefix = torch.cat([kept, token.to(device)[..., None]], -1)
            phi, G = phi_child.flatten(1).gather(1, top.indices), top.values
        if img_ids is not None:
            self._evict()
        if beam_node.shape[1] > S:
            kappa = G[:, S:S + 1]
            pi = -torch.expm1(-torch.exp(phi[:, :S] - kappa))
            w = torch.where(pi > 0, phi[:, :S].exp() / pi.clamp_min(1e-300), 0.0)
            beam_prefix = beam_prefix[:, :S]
        else:
            w = phi.exp()
        return beam_prefix, w.float().to(device)

    def entries(self, x_all, depth, S, generator, train=True, chunk=FORWARD_CHUNK):
        """Flattened (image row, prefix, weight) for every image in x_all -> img [E], prefix [E, depth], w [E]."""
        imgs, prefixes, weights = [], [], []
        for start in range(0, len(x_all), chunk):
            idx = torch.arange(start, min(start + chunk, len(x_all)), device=x_all.device)
            prefix, w = self.sample_distinct(x_all[idx], depth, S, generator, idx if train else None)
            imgs.append(idx.repeat_interleave(prefix.shape[1]))
            prefixes.append(prefix.reshape(prefix.shape[0] * prefix.shape[1], depth))
            weights.append(w.flatten())
        return torch.cat(imgs), torch.cat(prefixes), torch.cat(weights)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--vocab-size", type=int, default=2)
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--folds", type=int, default=8)
    p.add_argument("--num-candidates", type=int, default=64, help="candidate partitions generated per position")
    p.add_argument("--keep-candidates", type=int, default=32, help="best candidates (by in-sample objective) trained")
    p.add_argument("--prefix-samples", type=int, default=4, help="S distinct prefixes per image for fitting/scoring")
    p.add_argument("--labels", choices=["aligned", "random"], default="aligned",
                   help="label codewords by rank along a shared direction, or shuffle them per group")
    p.add_argument("--dedup-ari", type=float, default=0.95, help="drop candidates whose partition matches a better one")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=20, help="max epochs per position")
    p.add_argument("--patience", type=int, default=3)
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
    X = to_float(load_mnist("train", args.data_root, device)[: args.limit_train])
    Xv = to_float(load_mnist("val", args.data_root, device)[: args.limit_val])
    mean = X.flatten(1).mean(0)
    Xc, Xvc = X.flatten(1) - mean, Xv.flatten(1) - mean  # centered pixels, for the objective's precision
    N, Nv, V, Fo, S = len(X), len(Xv), args.vocab_size, args.folds, args.prefix_samples
    print(f"device {device}, {N} training images in {Fo} folds, {Nv} val images, V={V}, k_max={args.k_max}")

    state: dict[str, Any] = {"args": vars(args), "stages": [], "history": []}
    if args.resume:
        state = torch.load(args.state, map_location="cpu", weights_only=False)
        fold_of = state["fold_of"]
        print(f"resuming from {args.state}: {len(state['stages'])} positions built")
    else:
        fold_of = torch.randperm(N, generator=torch.Generator().manual_seed(args.seed)) % Fo
        state["fold_of"] = fold_of
    sampler = PrefixSampler(X, fold_of, V, Fo, args.memo_cap)
    stage_cfgs, final_states = [], []
    for t, st in enumerate(state["stages"]):
        cfg = st["cfg"]
        models = []
        for model_state in st["model_states"]:
            m = StageNet(V, t, cfg["n_heads"], cfg["width"]).to(device)
            m.load_state_dict(model_state)
            models.append(m)
        sampler.add_stage(models, Selector(cfg["sel_nodes"], cfg["sel_heads"], cfg["default_head"]))
        stage_cfgs.append(cfg)
        final_states.append(st["model_states"][Fo])

    # mean distance with no code, for reference: val image vs all training images
    k0 = ((Xvc.pow(2).sum(1) - 2 * Xvc @ Xc.mean(0) + Xc.pow(2).sum(1).mean()) / Xc.shape[1]).mean().item()
    all_idx = torch.arange(N, device=device)

    try:
        for t in range(len(state["stages"]), args.k_max):
            t0 = time.time()
            print(f"\n=== position {t + 1} of {args.k_max} (prefix length {t})")
            gen = torch.Generator().manual_seed(args.seed * 1000 + t)

            # 1-2. weighted prefix entries, candidates, pruning
            img, prefix, w = sampler.entries(X, t, S, gen)
            ids = node_ids(prefix, V)
            entries = Entries(Xc, img, ids, w, t, V)
            u = weighted_top_direction(entries, gen)
            base = fit_candidate(entries, 0, u, gen, args.labels)  # m = 0; also the fallback for unseen groups
            pool = [base] + [fit_candidate(entries, int(torch.randint(t + 1, (1,), generator=gen)), u, gen,
                                           args.labels, base) for _ in range(args.num_candidates - 1)]
            labels = [rule.labels(Xc, img, ids) for rule in pool]
            proxies = [candidate_proxy(entries, lab, N) for lab in labels]
            kept: list[int] = []
            for i in sorted(range(len(pool)), key=lambda j: proxies[j]):
                if all(weighted_node_ari(labels[i], labels[j], ids, w, V) <= args.dedup_ari for j in kept):
                    kept.append(i)
                    if len(kept) == args.keep_candidates:
                        break
            rules = [pool[i] for i in kept]
            heads = [f"{pool[i].name}#{i}" for i in kept]
            print(f"  {len(img)} prefix entries over {len(ids.unique())} prefixes; {len(pool)} candidates "
                  f"(m: {dict(sorted(Counter(r.m for r in pool).items()))}), kept {len(rules)} after dedup/pruning; "
                  f"in-sample objective of kept: {min(proxies[i] for i in kept):.5f}..{max(proxies[i] for i in kept):.5f}")

            # 3-4. multi-head training on folds, held-out scoring and per-prefix head selection
            models = [StageNet(V, t, len(rules), args.width).to(device) for _ in range(Fo + 1)]
            opts = [torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.weight_decay) for m in models]
            train_sets = [all_idx[(fold_of != f).to(device)] for f in range(Fo)] + [all_idx]
            entry_fold = fold_of[img.cpu()]
            best: dict[str, Any] = {"score": math.inf}
            bad = 0
            for epoch in range(1, args.epochs + 1):
                last_log, losses = time.time(), []
                for m, (model, opt) in enumerate(zip(models, opts)):
                    model.train()
                    order = train_sets[m][torch.randperm(len(train_sets[m]), generator=gen).to(device)]
                    for b in order.split(args.batch_size):
                        bprefix = sampler.sample_train(b, t, gen)
                        bids = node_ids(bprefix, V)
                        targets = torch.stack([r.labels(Xc, b, bids) for r in rules], 1)
                        loss = F.cross_entropy(model(X[b], bprefix).reshape(-1, V), targets.reshape(-1))
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                        losses.append(loss.item())
                        if time.time() - last_log >= args.log_interval:
                            last_log = time.time()
                            print(f"  epoch {epoch} model {m + 1}/{Fo + 1} cross-entropy "
                                  f"{sum(losses[-50:]) / len(losses[-50:]):.4f} (memo {len(sampler.memo)})", flush=True)

                with torch.no_grad():
                    probs = torch.empty(len(img), len(rules), V, device=device)
                    for f in range(Fo):
                        rows = (entry_fold == f).nonzero().squeeze(1).to(device)
                        models[f].eval()
                        probs[rows] = forward_chunked(models[f], X[img[rows]], prefix[rows]).softmax(-1)
                    rows_all = torch.arange(len(img), device=device)
                    costs = torch.stack([node_costs(Xc, img, ids, probs[:, h], Xc, img, ids, probs[:, h], w, rows_all)
                                         for h in range(len(rules))], 1)
                    groups, group_costs = costs_by_group(ids, costs * w[:, None])
                    hybrid = group_costs.min(1).values.sum().item() / N
                    per_head = (group_costs.sum(0) / N).tolist()
                improved = hybrid < best["score"]
                top3 = sorted(zip(per_head, heads))[:3]
                print(f"  epoch {epoch}: train cross-entropy {sum(losses) / len(losses):.4f} | held-out objective, "
                      f"best head per prefix {hybrid:.5f}; best single heads {[(h, round(c, 5)) for c, h in top3]}"
                      f"{'  *best*' if improved else ''}", flush=True)
                if improved:
                    best = {"score": hybrid, "epoch": epoch, "per_head": per_head, "groups": groups,
                            "choice": group_costs.argmin(1), "default": int(torch.tensor(per_head).argmin()),
                            "probs": probs.clone(), "states": [copy.deepcopy(m.state_dict()) for m in models]}
                    bad = 0
                else:
                    bad += 1
                    if bad >= args.patience:
                        break

            for m, s in zip(models, best["states"]):
                m.load_state_dict(s)
            selector = Selector(best["groups"].cpu(), best["choice"].cpu(), best["default"])
            sampler.add_stage(models, selector)
            counts = torch.bincount(best["choice"].cpu(), minlength=len(rules)).tolist()
            cfg = dict(n_heads=len(rules), width=args.width, candidates=heads, sel_nodes=best["groups"].tolist(),
                       sel_heads=best["choice"].tolist(), default_head=best["default"])
            stage_cfgs.append(cfg)
            final_states.append(copy.deepcopy(models[Fo].state_dict()))

            # 5. honest score on the val split: val images coded by the final models (S distinct prefixes each),
            # training entries (coded out-of-fold) as references; per-prefix head selection vs one head everywhere.
            with torch.no_grad():
                val_gen = torch.Generator().manual_seed(args.seed + 7 + t)
                vimg, vprefix, vw = sampler.entries(Xv, t, S, val_gen, train=False)
                vids = node_ids(vprefix, V)
                vprobs = forward_chunked(models[Fo], Xv[vimg], vprefix).softmax(-1)
                no_ref = torch.full((len(vimg),), -1, dtype=torch.long, device=device)
                honest = {}
                for label, choose in [("per-prefix", selector),
                                      ("single head", lambda q: torch.full_like(q, best["default"]))]:
                    ha, hr = choose(vids), choose(ids)
                    pa = vprobs[torch.arange(len(vimg), device=device), ha]
                    pr = best["probs"][torch.arange(len(img), device=device), hr]
                    cost = node_costs(Xvc, vimg, vids, pa, Xc, img, ids, pr, w, no_ref)
                    honest[label] = float((vw * cost).sum() / Nv)
            chosen = {heads[h]: c for h, c in enumerate(counts) if c}
            entry = {"position": t + 1, "epochs": best["epoch"], "candidates": heads, "heads_chosen": chosen,
                     "held_out": best["score"], "held_out_per_head": dict(zip(heads, best["per_head"])),
                     "val_per_prefix": honest["per-prefix"], "val_single_head": honest["single head"],
                     "k0": k0, "seconds": time.time() - t0}
            state["history"].append(entry)
            print(f"  position {t + 1} done in {entry['seconds']:.0f}s (best epoch {best['epoch']}): heads chosen per prefix "
                  f"{chosen}; val objective at k={t + 1}: per-prefix {honest['per-prefix']:.5f}, single head "
                  f"({heads[best['default']]}) {honest['single head']:.5f}; no code {k0:.5f}")

            state["stages"].append({"cfg": cfg, "model_states": [copy.deepcopy(m.state_dict()) for m in models]})
            torch.save(state, args.state)
            encoder = StagedEncoder(V, len(stage_cfgs), stage_cfgs)
            for net, s in zip(encoder.nets, final_states):
                net.load_state_dict(s)
            save_checkpoint(args.out, "encoder", encoder, args=vars(args), history=state["history"])
            print(f"  saved {args.out} ({len(stage_cfgs)} positions) and {args.state}")
    except KeyboardInterrupt:
        print(f"\ninterrupted; {len(state['stages'])} completed positions are in {args.state} (continue with --resume)")


if __name__ == "__main__":
    main()
