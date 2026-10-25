"""Stage 1: build the blind knight encoder (dvae/knight.py) round by round (see README).

A table holds every slice's held-out output on every training image (each image through the fold copy
that never trained on it), so any code's posterior over the training images is a product of table entries.
Each round:
  A. Codes. Sample C training images and an out-of-fold code for each, to depth T. At every distinct prefix,
     the bar: the bank's best realized gain among the slices not yet used along it.
  B. Partitions. Sample candidates*B distinct prefixes in proportion to their weight in J. For each, split its
     posterior as well as possible (Hartigan's method from a weighted k-means++ start, dvae/partition.py), then
     label every image by its nearest class centroid. Keep the B with the largest improvement J over all
     examined prefixes against the bar.
  C. Column. Train a new SliceNet column on the partitions (F+1 fold copies, lockstep epochs, early stopping
     on held-out cross-entropy) and add its held-out outputs to the table.
  D. Selector. Fit a fresh selector depth by depth on a fresh sample of C codes, early-stopped on the held-out
     gain of its choices; codes stop at depth T where the bank is used up or every prefix holds one image.
  E. Report the val decompression loss.

Saves the encoder after every round (--out, usable by the decoder/prior/QA scripts) and the full build
state (--state, every fold copy) so --resume continues with the next round.
"""

import argparse
import copy
import math
import os
import time
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F

from dvae.common import add_common_args, get_device, save_checkpoint
from dvae.data import load_mnist, to_float
from dvae.knight import BlindKnightEncoder, SelectorNet, SliceNet
from dvae.partition import CodeSet, PartitionSearch, kmeanspp

FORWARD_CHUNK = 2048  # images per forward pass when filling the table


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Progress:
    """Throttled progress lines inside a long step: at most one per interval seconds, all in one format."""

    def __init__(self, interval):
        self.interval, self.last = interval, time.time()

    def __call__(self, msg):
        if time.time() - self.last >= self.interval:
            self.last = time.time()
            log(f"    ... {msg}")


@contextmanager
def step(name, summary):
    """Run one step of a round, then log one line: its name, the summary it filled in, and its duration."""
    t0 = time.time()
    yield
    log(f"  {name}: {summary.get('text', '')} ({time.time() - t0:.1f}s)")


def fmt(values, digits=4):
    return " ".join(f"{v:.{digits}f}" for v in values)


@torch.no_grad()
def slice_table(copies, fold_of, X, Xv):
    """Every slice's output on every training image through its held-out copy [N, S, V], and on every val
    image through the final copy [Nv, S, V]."""
    for c in copies:
        c.eval()
    out = torch.empty(len(X), copies[0].n_slices, copies[0].V, device=X.device)
    for f in range(len(copies) - 1):
        rows = (fold_of == f).nonzero().squeeze(1)
        for i in range(0, len(rows), FORWARD_CHUNK):
            r = rows[i:i + FORWARD_CHUNK]
            out[r] = copies[f](X[r]).softmax(-1)
    val = torch.cat([copies[-1](Xv[i:i + FORWARD_CHUNK]).softmax(-1) for i in range(0, len(Xv), FORWARD_CHUNK)])
    return out, val


@torch.no_grad()
def extend(codes: CodeSet, selector, gen):
    """One more token for every sampled code: the selector picks an unused slice, the image answers it."""
    h = selector.pick(codes.tokens, codes.slices)
    probs = codes.table[codes.images, h]
    tokens = torch.multinomial(probs.cpu(), 1, generator=gen).squeeze(1).to(probs.device)
    codes.add_depth(h, tokens)


def fit_copies(modules, batches, loss_fn, score, epochs, patience, lr, weight_decay, progress, describe):
    """Train one module per copy in lockstep epochs, keeping every copy's weights from the epoch with the best
    held-out score (higher is better). batches(m) yields copy m's minibatches; loss_fn(m, batch) is its loss;
    describe(epoch, score) is the progress line. -> (best score, best epoch)."""
    opts = [torch.optim.AdamW([p for p in mod.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
            for mod in modules]
    best: dict[str, Any] = {"score": -math.inf, "epoch": 0, "states": [copy.deepcopy(m.state_dict()) for m in modules]}
    bad = 0
    for epoch in range(1, epochs + 1):
        for m, opt in enumerate(opts):
            modules[m].train()
            for batch in batches(m):
                loss = loss_fn(m, batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        for mod in modules:
            mod.eval()
        with torch.no_grad():
            s = score()
        progress(describe(epoch, s))
        if s > best["score"]:
            best = {"score": s, "epoch": epoch, "states": [copy.deepcopy(m.state_dict()) for m in modules]}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    for mod, st in zip(modules, best["states"]):
        mod.load_state_dict(st)
        mod.eval()
    return best["score"], best["epoch"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--vocab-size", type=int, default=16)
    p.add_argument("--k-max", type=int, default=8, help="code length the exported encoder gives the decoder and prior")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--round-patience", type=int, default=3, help="stop after this many rounds without a better val loss")
    p.add_argument("--slices-per-round", type=int, default=8, help="B: new partitions (one column) per round")
    p.add_argument("--codes", type=int, default=512, help="C: sampled training codes examined per step")
    p.add_argument("--weight-cutoff", type=float, default=1e-4, help="posterior weights below this are dropped")
    p.add_argument("--k-decay", type=float, default=0.8, help="rho: each further token is read with this probability")
    p.add_argument("--candidates", type=int, default=8, help="candidate partitions per kept one: B * this many codes")
    p.add_argument("--min-support", type=int, default=100, help="codes need this much posterior weight (expected images) to be split")
    p.add_argument("--hartigan-rounds", type=int, default=100, help="cap on batched Hartigan move rounds per start")
    p.add_argument("--hartigan-tol", type=float, default=1e-4,
                   help="stop Hartigan when J rises by less than this fraction per round, over the last 5 rounds")
    p.add_argument("--column-width", type=int, default=32)
    p.add_argument("--column-hidden", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=20, help="max column epochs per round")
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--selector-width", type=int, default=128)
    p.add_argument("--selector-batch-size", type=int, default=256)
    p.add_argument("--selector-lr", type=float, default=1e-3)
    p.add_argument("--selector-epochs", type=int, default=100, help="max selector epochs per depth")
    p.add_argument("--selector-patience", type=int, default=5)
    p.add_argument("--log-interval", type=float, default=10.0, help="seconds between progress lines")
    p.add_argument("--limit-train", type=int, default=None, help="use only the first N training images (smoke tests)")
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--out", default="checkpoints/encoder.pt")
    p.add_argument("--state", default="checkpoints/encoder_build.pt")
    p.add_argument("--resume", action="store_true", help="continue from --state with the next round")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    X = to_float(load_mnist("train", args.data_root, device)[: args.limit_train])
    Xv = to_float(load_mnist("val", args.data_root, device)[: args.limit_val])
    mean = X.flatten(1).mean(0)
    Xc, Xvc = X.flatten(1) - mean, Xv.flatten(1) - mean  # centered pixels, for the objective's precision
    N, Nv, V, Fo, B, C = len(X), len(Xv), args.vocab_size, args.folds, args.slices_per_round, args.codes
    P, rho, cutoff = Xc.shape[1], args.k_decay, args.weight_cutoff
    gen = torch.Generator().manual_seed(args.seed)
    progress = Progress(args.log_interval)

    def new_copy():
        return SliceNet(V, (), args.column_width, args.column_hidden).to(device)

    state: dict[str, Any] = {"args": vars(args), "history": []}
    copies = [new_copy() for _ in range(Fo + 1)]
    selector: SelectorNet | None = None
    T, first_round = 0, 1
    if args.resume and os.path.exists(args.state):
        state = torch.load(args.state, map_location="cpu", weights_only=False)
        fold_of = state["fold_of"].to(device)
        for c, st in zip(copies, state["copies"]):
            for cfg in state["columns"]:
                c.add_column(cfg["n_slices"], device)
            c.load_state_dict(st)
        selector = SelectorNet(V, copies[0].n_slices, args.selector_width).to(device)
        selector.load_state_dict(state["selector"])
        T, first_round = state["T"], len(state["history"]) + 1
    else:
        if args.resume:
            log(f"--resume: no build state at {args.state}, starting from round 1")
        fold_of = (torch.randperm(N, generator=gen) % Fo).to(device)
        state["fold_of"] = fold_of.cpu()
    for c in copies:
        c.requires_grad_(False)
    table, val_table = slice_table(copies, fold_of, X, Xv)
    held_out = [(fold_of == f).nonzero().squeeze(1) for f in range(Fo)]
    log(f"device {device}, {N} training images in {Fo} folds, {Nv} val images, V={V}, {B} slices per round, "
        f"{C} codes per step, rho={rho}" + (f"; resuming after round {first_round - 1}" if first_round > 1 else ""))

    best_val, bad_rounds = min([h["val_loss"] for h in state["history"]], default=math.inf), 0
    try:
        for rnd in range(first_round, args.rounds + 1):
            S = copies[0].n_slices
            log(f"round {rnd}: {S} slices in the bank, codes reach depth {T}")
            entry: dict[str, Any] = {"round": rnd}

            # A. a sample of the encoder's codes, and the bank's best realized gain at each prefix
            summary: dict[str, str] = {}
            with step("A codes", summary), torch.no_grad():
                codes = CodeSet(table, Xc, cutoff, torch.randperm(N, generator=gen)[:C].to(device))
                for _ in range(T):
                    extend(codes, selector, gen)
                all_codes = torch.arange(codes.n_codes, device=device)
                G = codes.slice_gains(all_codes, torch.arange(S, device=device),
                                      lambda k, n: progress(f"A bank gains: prefix {k}/{n}"))
                G = G.masked_fill(codes.used_mask(all_codes, S), -torch.inf)
                bar = G.max(1).values.clamp_min(0) if S else torch.zeros(codes.n_codes, device=device)
                a = rho ** codes.depth.float() * codes.count / C
                summary["text"] = (f"{codes.n_codes} distinct prefixes from {C} samples, "
                                   f"bank's weighted gain {float((a * bar).sum()):.5f}")

            # B. candidate partitions, one per sampled code, each splitting that code's posterior as well as it can;
            # keep the B that most improve the bank overall
            summary = {}
            with step("B partitions", summary), torch.no_grad():
                search = PartitionSearch(codes, a, V)
                bar_active = bar[search.code_ids]
                eligible = ((codes.W >= args.min_support) & (codes.support >= 2)).nonzero().squeeze(1)
                n_cand = min(args.candidates * B, len(eligible))
                aims = eligible[torch.multinomial(a[eligible].cpu(), n_cand, generator=gen).to(device)] if n_cand else eligible
                cands: list[tuple[float, int, torch.Tensor]] = []
                rounds = []
                for k, z in enumerate(aims.tolist()):
                    one = torch.zeros(codes.n_codes, device=device)
                    one[z] = 1.0
                    local = PartitionSearch(codes, one, V, only=torch.tensor([z], device=device))
                    img, w = codes.posterior(z)
                    weight = torch.zeros(N, device=device).index_add_(0, img, w)
                    where = f"B candidate {k + 1}/{n_cand} (depth {int(codes.depth[z])})"
                    lab, _, r = local.run(kmeanspp(Xc, V, gen, weight), torch.zeros(1, device=device),
                                          args.hartigan_rounds, args.hartigan_tol,
                                          progress=lambda r, J, where=where: progress(f"{where}, Hartigan round {r}"))
                    rounds.append(r)
                    lab = local.centroid_labels(lab)
                    cands.append((search.objective(search.gains(search.stats(lab)), bar_active), z, lab))
                cands.sort(key=lambda c: c[0], reverse=True)
                kept = [c for c in cands[:B] if c[0] > 0]  # a partition that beats the bar nowhere would teach nothing
                Bk = len(kept)
                labels = torch.stack([c[2] for c in kept]) if kept else torch.empty(0, N, dtype=torch.long, device=device)
                nu = torch.zeros(N, Bk, dtype=torch.bool, device=device)
                for b, (_, _, lab) in enumerate(kept):
                    gains = search.gains(search.stats(lab))
                    beats = gains[search.e_code] > bar_active[search.e_code]
                    nu[search.e_img[beats], b] = True
                implied = [c[0] for c in kept]
                where_kept = [(int(codes.depth[c[1]]), round(float(codes.W[c[1]]))) for c in kept]
                summary["text"] = (f"{n_cand} candidates from {len(eligible)} eligible codes, "
                                   f"{sum(rounds) / max(len(rounds), 1):.1f} Hartigan rounds each; kept "
                                   + ", ".join(f"depth {d} ({s} images) J {J:.5f}"
                                               for (d, s), J in zip(where_kept, implied)))
            entry.update(implied_J=implied, targets=where_kept)
            if Bk == 0:
                log("stopping: no candidate partition beats the bank anywhere")
                break

            # C. a new column learns the partitions; earlier columns stay frozen
            summary = {}
            with step("C column", summary):
                columns = [c.add_column(Bk, device) for c in copies]
                labeled = nu.any(1).nonzero().squeeze(1)
                weights = nu.float()

                def ce(m, idx):
                    logits = copies[m](X[idx])[:, S:]  # [n, Bk, V]
                    loss = F.cross_entropy(logits.reshape(-1, V), labels[:, idx].T.reshape(-1), reduction="none")
                    w = weights[idx].reshape(-1)
                    return (loss * w).sum() / w.sum().clamp_min(1)

                def ce_batches(m):
                    rows = labeled if m == Fo else labeled[fold_of[labeled] != m]
                    order = rows[torch.randperm(len(rows), generator=gen).to(device)]
                    return order.split(args.batch_size)

                def held_out_ce():
                    total, count = 0.0, 0.0
                    for f in range(Fo):
                        rows = labeled[fold_of[labeled] == f]
                        for idx in rows.split(FORWARD_CHUNK):
                            total += float(ce(f, idx)) * float(weights[idx].sum())
                            count += float(weights[idx].sum())
                    return -total / max(count, 1.0)

                score, epoch = fit_copies(columns, ce_batches, ce, held_out_ce, args.epochs, args.patience, args.lr,
                                          args.weight_decay, progress,
                                          lambda e, s: f"C epoch {e}: held-out cross-entropy {-s:.4f}")
                for c in copies:
                    c.requires_grad_(False)
                table, val_table = slice_table(copies, fold_of, X, Xv)
                codes.table = table
                new = torch.arange(S, S + Bk, device=device)
                realized_gains = codes.slice_gains(search.code_ids, new,
                                                   lambda k, n: progress(f"C realized gains: prefix {k}/{n}"))
                realized = [search.objective(realized_gains[:, b], bar_active) for b in range(Bk)]
                S += Bk
                summary["text"] = f"best epoch {epoch}, held-out cross-entropy {-score:.4f}, realized J {fmt(realized, 5)}"
            entry.update(column_epochs=epoch, held_out_cross_entropy=-score, realized_J=realized)

            # D. a fresh selector learns, depth by depth, which slice gains most after each prefix
            summary = {}
            with step("D selector", summary):
                sels = [SelectorNet(V, S, args.selector_width).to(device) for _ in range(Fo + 1)]
                codes = CodeSet(table, Xc, cutoff, torch.randperm(N, generator=gen)[:C].to(device))
                sample_fold = fold_of[codes.images]
                targets: list[torch.Tensor] = []  # per depth: every sample's gain per slice [C, S]
                masks: list[torch.Tensor] = []  # per depth: True for slices already used [C, S]
                drift: list[float] = []
                scale, choice = 1.0, 0.0
                for u in range(S + 1):
                    if u >= 1:
                        extend(codes, sels[Fo], gen)
                    here = codes.codes_at(u)
                    if u == S or bool((codes.support[here] < 2).all()):
                        T = u
                        break
                    with torch.no_grad():
                        G = codes.slice_gains(here, torch.arange(S, device=device),
                                              lambda k, n: progress(f"D gains at depth {u}: prefix {k}/{n}"))
                        local = codes.ids[:, u] - here[0]
                        targets.append(G[local])
                        masks.append(codes.used_mask(codes.ids[:, u], S))
                    if u == 0:
                        scale = float(targets[0].abs().mean()) or 1.0
                        for s in sels:
                            with torch.no_grad():
                                s.out.bias.copy_(targets[0][0] / scale)
                    depths = list(range(u + 1))

                    def sel_batches(m, depths=depths):
                        out = []
                        for d in depths:
                            rows = torch.arange(C, device=device)
                            rows = rows if m == Fo else rows[sample_fold != m]
                            order = rows[torch.randperm(len(rows), generator=gen).to(device)]
                            out += [(d, idx) for idx in order.split(args.selector_batch_size)]
                        return [out[i] for i in torch.randperm(len(out), generator=gen).tolist()]

                    def mse(m, batch):
                        d, idx = batch
                        pred = sels[m](codes.tokens[idx, :d])
                        free = ~masks[d][idx]
                        return ((pred - targets[d][idx] / scale).pow(2) * free).sum() / free.sum().clamp_min(1)

                    def choices(depths=depths):
                        total = 0.0
                        for d in depths:
                            got = torch.zeros(C, device=device)
                            for f in range(Fo):
                                rows = (sample_fold == f).nonzero().squeeze(1)
                                pick = sels[f](codes.tokens[rows, :d]).masked_fill(masks[d][rows], -torch.inf).argmax(1)
                                got[rows] = targets[d][rows, pick]
                            total += rho**d * float(got.mean())
                        return total

                    choice, _ = fit_copies(sels, sel_batches, mse, choices, args.selector_epochs, args.selector_patience,
                                           args.selector_lr, 0.0, progress,
                                           lambda e, s, u=u: f"D depth {u}, epoch {e}: held-out gain of choices {s:.5f}")
                    for s in sels[:Fo]:
                        s.load_state_dict(sels[Fo].state_dict())
                    with torch.no_grad():
                        if u >= 1:
                            path = sels[Fo].path(codes.tokens[:, :u])
                            drift = (path != codes.slices[:, :u]).float().mean(0).tolist()
                selector = sels[Fo]
                used = codes.slices.unique().numel() if T else 0
                summary["text"] = (f"codes stop at depth {T}, held-out gain of choices {choice:.5f}, "
                                   f"{used} of {S} slices used" + (f", drift {fmt(drift, 2)}" if drift else ""))
            entry.update(T=T, selector_choice_gain=choice, slices_used=used, drift=drift)

            # E. score on held-out images
            summary = {}
            with step("E val", summary), torch.no_grad():
                assert selector is not None
                vgen = torch.Generator().manual_seed(args.seed + 7)
                vtok = torch.zeros(Nv, 0, dtype=torch.long, device=device)
                vsl = vtok.clone()
                for _ in range(T):
                    h = selector.pick(vtok, vsl)
                    probs = val_table[torch.arange(Nv, device=device), h]
                    t = torch.multinomial(probs.cpu(), 1, generator=vgen).to(device)
                    vtok, vsl = torch.cat([vtok, t], 1), torch.cat([vsl, h[:, None]], 1)
                sq, vsq = Xc.pow(2).sum(1), Xvc.pow(2).sum(1)
                L = torch.zeros(T + 1, device=device)
                rows_per = max(1, 2**24 // N)
                for i in range(0, Nv, rows_per):
                    v = slice(i, i + rows_per)
                    w = torch.ones(len(vsq[v]), N, device=device)
                    alive = torch.ones(len(w), dtype=torch.bool, device=device)
                    last = torch.zeros(len(w), device=device)
                    for t in range(T + 1):
                        if t > 0:
                            w = w * table[:, vsl[v, t - 1], vtok[v, t - 1]].T
                            w = torch.where(w >= cutoff, w, 0.0)
                        W = w.sum(1)
                        dist = (W * vsq[v] - 2 * (Xvc[v] * (w @ Xc)).sum(1) + w @ sq) / (P * W.clamp_min(1e-30))
                        alive &= W > 0  # past the longest prefix with weight, the loss stays where it was
                        last = torch.where(alive, dist, last)
                        L[t] += last.sum()
                L = (L / Nv).tolist()
                val_loss = sum((1 - rho) * rho ** (t - 1) * L[t] for t in range(1, T + 1)) + rho**T * L[T]
                summary["text"] = f"loss {val_loss:.5f} (no code {L[0]:.5f}; by length {fmt(L[1:], 5)})"
            entry.update(val_loss=val_loss, val_loss_by_length=L)
            state["history"].append(entry)

            # save the encoder (final copy and selector) and the build state
            assert selector is not None
            encoder = BlindKnightEncoder(V, min(args.k_max, S), copies[Fo].config, args.column_width, args.column_hidden,
                                         args.selector_width)
            encoder.slices.load_state_dict(copies[Fo].state_dict())
            encoder.selector.load_state_dict(selector.state_dict())
            save_checkpoint(args.out, "encoder", encoder.to(device), args=vars(args), history=state["history"])
            state.update(copies=[c.state_dict() for c in copies], columns=copies[0].config,
                         selector=selector.state_dict(), T=T)
            torch.save(state, args.state)

            if val_loss < best_val:
                best_val, bad_rounds = val_loss, 0
            else:
                bad_rounds += 1
                if bad_rounds >= args.round_patience:
                    log(f"stopping: no better val loss for {bad_rounds} rounds (best {best_val:.5f})")
                    break
    except KeyboardInterrupt:
        log(f"interrupted; {len(state['history'])} finished rounds are in {args.state} (continue with --resume)")


if __name__ == "__main__":
    main()
