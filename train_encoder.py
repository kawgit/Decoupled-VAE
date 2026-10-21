"""Stage 1: build the staged encoder (dvae/staged.py) one code position at a time.

For each position t = 0 .. k_max-1:
  1. Sample a prefix z_1..z_t for every training image from the out-of-fold models of earlier stages.
  2. Generate candidate partitions for the next token (dvae/partition.py) and deduplicate them.
  3. Train num_folds + 1 copies of a StageNet with one head per candidate, by cross-entropy against the
     candidates' labels: fold model f on the images outside fold f, the final model on all of them.
     Prefixes are resampled from the out-of-fold models every batch, so each stage learns to cope with
     the mistakes earlier stages really make on unseen images. Those distributions are memoized.
  4. After every epoch, score every head on each fold's held-out images with the decompression
     objective (exact, all training images as references), pick the best head per prefix, and keep the
     epoch with the best total. The decision is shared by all copies, so the final model is trained
     for the same number of epochs as the fold models that were scored.
  5. The final model with that per-prefix head selection becomes stage t. Its honest score is reported
     on the separate val split, for the per-prefix selection and for the single best head.

Saves the encoder after every stage (--out, usable by the decoder/prior/QA scripts) and the full build
state (--state, every fold model) so --resume can continue with the next stage.
"""

import argparse
import copy
import math
import time
from typing import Any

import torch
import torch.nn.functional as F

from dvae.common import add_common_args, get_device, save_checkpoint
from dvae.data import batches, load_mnist, to_float
from dvae.partition import CANDIDATES, costs_by_group, dedup, make_candidates, node_costs
from dvae.staged import Selector, StagedEncoder, StageNet, node_ids


FORWARD_CHUNK = 2048  # images per forward pass when scoring or sampling


def forward_chunked(model, x, prefix):
    """model(x, prefix) in chunks -> logits [N, n_heads, V]."""
    return torch.cat([model(x[i:i + FORWARD_CHUNK], prefix[i:i + FORWARD_CHUNK]) for i in range(0, len(x), FORWARD_CHUNK)])


class PrefixSampler:
    """Samples prefixes through the stages built so far.

    Training images go through the out-of-fold model of their own fold, and the next-token distribution
    of every (image, prefix) visited is memoized: stages are frozen once built, so it never changes. The
    memo holds at most memo_cap entries; when full it evicts the prefixes least likely to be sampled.
    Other images (val) go through the final models.
    """

    def __init__(self, images, fold_of, vocab_size, n_folds, memo_cap):
        self.images, self.fold_of, self.V, self.n_folds = images, fold_of, vocab_size, n_folds
        self.memo_cap, self.memo, self.stages = memo_cap, {}, []

    def add_stage(self, models, selector):
        for m in models:
            m.eval().requires_grad_(False)
        self.stages.append((models, selector))

    def _probs(self, t, x, prefix, model_idx):
        """Next-token distributions [n, V] (on CPU) from stage t's model_idx[i] model."""
        models, selector = self.stages[t]
        out = torch.empty(len(x), self.V)
        for m in model_idx.unique().tolist():
            rows = (model_idx == m).nonzero().squeeze(1)
            rows_dev = rows.to(x.device)
            logits = forward_chunked(models[m], x[rows_dev], prefix[rows_dev])
            head = selector(node_ids(prefix[rows_dev], self.V))
            out[rows] = logits[torch.arange(len(rows_dev), device=x.device), head].softmax(-1).cpu()
        return out

    @torch.no_grad()
    def sample_train(self, idx, depth, generator):
        """Prefixes [B, depth] for training images idx, through out-of-fold models (memoized)."""
        B, device = len(idx), idx.device
        prefix = torch.zeros(B, 0, dtype=torch.long, device=device)
        ids = idx.tolist()
        folds = self.fold_of[idx.cpu()]
        node = torch.zeros(B, dtype=torch.long)
        path = torch.ones(B, dtype=torch.float64)
        for t in range(depth):
            keys = list(zip(ids, [t] * B, node.tolist()))
            probs = torch.empty(B, self.V)
            hits, hit_vals, misses = [], [], []
            for b, key in enumerate(keys):
                value = self.memo.get(key)
                if value is None:
                    misses.append(b)
                else:
                    hits.append(b)
                    hit_vals.append(value[0])
            if hits:
                probs[hits] = torch.tensor(hit_vals)
            if misses:
                rows = torch.tensor(misses)
                rows_dev = rows.to(device)
                p = self._probs(t, self.images[idx[rows_dev]], prefix[rows_dev], folds[rows])
                probs[rows] = p
                for j, b in enumerate(misses):
                    self.memo[keys[b]] = (tuple(p[j].tolist()), float(path[b]))
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            path = path * probs[torch.arange(B), nxt].double()
            node = node * self.V + nxt
            prefix = torch.cat([prefix, nxt.to(device)[:, None]], 1)
        if len(self.memo) > self.memo_cap:  # evict the least likely prefixes
            keep = sorted(self.memo.items(), key=lambda item: item[1][1], reverse=True)[: int(0.9 * self.memo_cap)]
            self.memo = dict(keep)
        return prefix

    @torch.no_grad()
    def sample_final(self, x, depth, generator):
        """Prefixes [N, depth] for other images, through the final models."""
        prefix = torch.zeros(len(x), 0, dtype=torch.long, device=x.device)
        final = torch.full((len(x),), self.n_folds, dtype=torch.long)
        for t in range(depth):
            nxt = torch.multinomial(self._probs(t, x, prefix, final), 1, generator=generator).squeeze(1)
            prefix = torch.cat([prefix, nxt.to(x.device)[:, None]], 1)
        return prefix

    def sample_all_train(self, depth, generator, chunk=FORWARD_CHUNK):
        n = len(self.images)
        idx = torch.arange(n, device=self.images.device)
        return torch.cat([self.sample_train(idx[i:i + chunk], depth, generator) for i in range(0, n, chunk)]) \
            if depth > 0 else torch.zeros(n, 0, dtype=torch.long, device=self.images.device)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--vocab-size", type=int, default=2)
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--folds", type=int, default=8)
    p.add_argument("--candidates", default=",".join(CANDIDATES), help=f"comma-separated subset of {CANDIDATES}")
    p.add_argument("--dedup-ari", type=float, default=0.95, help="drop candidates whose partition matches an earlier one")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=20, help="max epochs per stage")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--memo-cap", type=int, default=2_000_000)
    p.add_argument("--log-interval", type=float, default=5.0, help="seconds between progress lines")
    p.add_argument("--limit-train", type=int, default=None, help="use only the first N training images (smoke tests)")
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--out", default="checkpoints/encoder.pt")
    p.add_argument("--state", default="checkpoints/encoder_build.pt")
    p.add_argument("--resume", action="store_true", help="continue from --state with the next stage")
    args = p.parse_args()
    names = [n.strip() for n in args.candidates.split(",") if n.strip()]
    assert all(n in CANDIDATES for n in names), f"unknown candidate in {names}"

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    train_u8 = load_mnist("train", args.data_root, device)[: args.limit_train]
    val_u8 = load_mnist("val", args.data_root, device)[: args.limit_val]
    X, Xv = to_float(train_u8), to_float(val_u8)
    mean = X.flatten(1).mean(0)
    Xc, Xvc = X.flatten(1) - mean, Xv.flatten(1) - mean  # centered pixels, for the objective's precision
    N, V, Fo = len(X), args.vocab_size, args.folds
    all_idx = torch.arange(N, device=device)
    print(f"device {device}, {N} training images in {Fo} folds, {len(Xv)} val images, V={V}, k_max={args.k_max}")

    state = {"args": vars(args), "stages": [], "history": []}
    if args.resume:
        state = torch.load(args.state, map_location="cpu", weights_only=False)
        fold_of = state["fold_of"]
        print(f"resuming from {args.state}: {len(state['stages'])} stages built")
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
    sq_tr = Xc.pow(2).sum(1)
    k0 = ((Xvc.pow(2).sum(1) - 2 * Xvc @ Xc.mean(0) + sq_tr.mean()) / Xc.shape[1]).mean().item()

    try:
        for t in range(len(state["stages"]), args.k_max):
            t0 = time.time()
            print(f"\n=== stage {t} (token {t + 1} of {args.k_max})")
            gen = torch.Generator().manual_seed(args.seed * 1000 + t)
            fit_prefix = sampler.sample_all_train(t, gen)  # one prefix per image: fits candidates, scores heads
            fit_ids = node_ids(fit_prefix, V)
            rules = dedup(make_candidates(Xc, fit_ids, V, names, gen), Xc, fit_ids, V, args.dedup_ari)
            heads = list(rules)
            print(f"  {len(fit_ids.unique())} prefixes in use; candidates: {heads}")

            models = [StageNet(V, t, len(heads), args.width).to(device) for _ in range(Fo + 1)]
            opts = [torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.weight_decay) for m in models]
            train_sets = [all_idx[(fold_of != f).to(device)] for f in range(Fo)] + [all_idx]
            best: dict[str, Any] = {"score": math.inf}
            bad = 0
            for epoch in range(1, args.epochs + 1):
                last_log, losses = time.time(), []
                for m, (model, opt) in enumerate(zip(models, opts)):
                    model.train()
                    order = train_sets[m][torch.randperm(len(train_sets[m]), generator=gen).to(device)]
                    for b in order.split(args.batch_size):
                        prefix = sampler.sample_train(b, t, gen)
                        ids = node_ids(prefix, V)
                        labels = torch.stack([rules[h].labels(Xc[b], ids) for h in heads], 1)
                        loss = F.cross_entropy(model(X[b], prefix).reshape(-1, V), labels.reshape(-1))
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                        losses.append(loss.item())
                        if time.time() - last_log >= args.log_interval:
                            last_log = time.time()
                            print(f"  epoch {epoch} model {m + 1}/{Fo + 1} cross-entropy {sum(losses[-50:]) / len(losses[-50:]):.4f} "
                                  f"(memo {len(sampler.memo)})", flush=True)

                # Score every head on each fold's held-out images, all training images as references.
                with torch.no_grad():
                    probs = torch.empty(N, len(heads), V, device=device)
                    for f in range(Fo):
                        rows = (fold_of == f).nonzero().squeeze(1).to(device)
                        models[f].eval()
                        probs[rows] = forward_chunked(models[f], X[rows], fit_prefix[rows]).softmax(-1)
                    costs = torch.stack([node_costs(Xc, fit_ids, probs[:, h], Xc, fit_ids, probs[:, h], all_idx)
                                         for h in range(len(heads))], 1)
                    groups, group_costs = costs_by_group(fit_ids, costs)
                    hybrid = group_costs.min(1).values.sum().item() / N
                    per_head = (group_costs.sum(0) / N).tolist()
                improved = hybrid < best["score"]
                print(f"  epoch {epoch}: train cross-entropy {sum(losses) / len(losses):.4f} | held-out objective, "
                      f"best head per prefix {hybrid:.5f}; per head {dict(zip(heads, [round(v, 5) for v in per_head]))}"
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
            counts = torch.bincount(best["choice"].cpu(), minlength=len(heads)).tolist()
            cfg = dict(n_heads=len(heads), width=args.width, candidates=heads, sel_nodes=best["groups"].tolist(),
                       sel_heads=best["choice"].tolist(), default_head=best["default"])
            stage_cfgs.append(cfg)
            final_states.append(copy.deepcopy(models[Fo].state_dict()))

            # Honest score on the val split: val images coded by the final models, training images (coded
            # out-of-fold) as references; per-prefix head selection vs the single best head everywhere.
            with torch.no_grad():
                val_gen = torch.Generator().manual_seed(args.seed + 7)
                val_prefix = sampler.sample_final(Xv, t, val_gen)
                val_ids = node_ids(val_prefix, V)
                val_probs = forward_chunked(models[Fo], Xv, val_prefix).softmax(-1)
                no_ref = torch.full((len(Xv),), -1, dtype=torch.long, device=device)
                honest = {}
                for label, choose in [("per-prefix", selector), ("single head", lambda ids: torch.full_like(ids, best["default"]))]:
                    ha, hr = choose(val_ids), choose(fit_ids)
                    pa = val_probs[torch.arange(len(Xv), device=device), ha]
                    pr = best["probs"][all_idx, hr]
                    honest[label] = node_costs(Xvc, val_ids, pa, Xc, fit_ids, pr, no_ref).mean().item()
            entry = {"stage": t, "epochs": best["epoch"], "candidates": heads, "heads_chosen": dict(zip(heads, counts)),
                     "held_out": best["score"], "held_out_per_head": dict(zip(heads, best["per_head"])),
                     "val_per_prefix": honest["per-prefix"], "val_single_head": honest["single head"],
                     "k0": k0, "seconds": time.time() - t0}
            state["history"].append(entry)
            print(f"  stage {t} done in {entry['seconds']:.0f}s (best epoch {best['epoch']}): heads chosen per prefix "
                  f"{entry['heads_chosen']}; val objective at k={t + 1}: per-prefix {honest['per-prefix']:.5f}, "
                  f"single head ({heads[best['default']]}) {honest['single head']:.5f}; no code {k0:.5f}")

            state["stages"].append({"cfg": cfg, "model_states": [copy.deepcopy(m.state_dict()) for m in models]})
            torch.save(state, args.state)
            encoder = StagedEncoder(V, len(stage_cfgs), stage_cfgs)
            for net, s in zip(encoder.nets, final_states):
                net.load_state_dict(s)
            save_checkpoint(args.out, "encoder", encoder, args=vars(args), history=state["history"])
            print(f"  saved {args.out} ({len(stage_cfgs)} stages) and {args.state}")
    except KeyboardInterrupt:
        print(f"\ninterrupted; {len(state['stages'])} completed stages are in {args.state} (continue with --resume)")


if __name__ == "__main__":
    main()
