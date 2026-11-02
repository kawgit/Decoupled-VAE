"""Stage 1: build the question-answer encoder (dvae/questions.py) round by round (see README).

A table holds every question's held-out answer distribution on every training image (each image through the fold
copy that never trained on it), so any code's posterior over the training images is a product of table entries.
Committed nodes always ask the question built for them; the selector picks everywhere else. Each round:
  A. Frontier. The uncommitted children of committed nodes (the root at first), scored by the loss their
     children would still hold under the question asked there now: rho^depth * images * (spread - its gain).
     The B highest are this round's targets.
  B. Partitions. For each target, the best V-means split of its posterior (Hartigan's method from weighted
     k-means++ starts, dvae/partition.py), extended to every image by nearest class centroid.
  C. Questions. Train one new question network per target on its partition, each image weighted by its
     posterior weight in the target (F+1 fold copies, all trained together in one fused pass, early stopping on
     held-out cross-entropy), add their held-out answers to the table, and commit each target to its new question.
  D. Selector. Fit a fresh selector depth by depth on a fresh sample of C codes, early-stopped on the held-out
     gain of its choices at uncommitted prefixes; codes stop at depth T where the bank is used up or every
     prefix holds one image.
  E. Report the val decompression loss.

Saves the encoder after every round (--out, usable by the decoder/prior/QA scripts) and the full build
state (--state, every fold copy) so --resume continues with the next round.
"""

import argparse
import copy
import math
import os
import textwrap
import time
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F

from dvae.common import add_common_args, get_device, save_checkpoint
from dvae.data import NUM_PIXELS, load_labels, load_mnist, to_float
from dvae.partition import CodeSet, PartitionSearch, kmeanspp, posterior_gains, spread
from dvae.plots import plot_build
from dvae.questions import QuestionAnswerEncoder, QuestionNets, SelectorNet, question_path, selector_pick

FORWARD_CHUNK = 2048  # images per forward pass when filling the table


WIDTH = 100  # longest log line, timestamp included


def log(msg):
    """Print each line of msg after a timestamp, wrapped at WIDTH; wrapped parts are indented a little further."""
    stamp = f"[{time.strftime('%H:%M:%S')}] "
    for line in msg.split("\n"):
        lead = " " * (len(line) - len(line.lstrip()))
        parts = textwrap.wrap(line.strip(), WIDTH - len(stamp), initial_indent=lead, subsequent_indent=lead + "    ",
                              break_on_hyphens=False) or [""]
        for i, part in enumerate(parts):
            print((stamp if i == 0 else " " * len(stamp)) + part, flush=True)


class Progress:
    """Progress lines inside a long step, one kind per loop. The first and the last line of each kind are always
    logged; in between, at most one line per interval seconds."""

    def __init__(self, interval):
        self.interval, self.last = interval, 0.0
        self.seen: set[str] = set()
        self.pending: dict[str, str] = {}  # each kind's latest line not logged yet

    def __call__(self, kind, msg):
        if kind not in self.seen or time.time() - self.last >= self.interval:
            self.seen.add(kind)
            self.pending.pop(kind, None)
            self._log(msg)
        else:
            self.pending[kind] = msg

    def _log(self, msg):
        self.last = time.time()
        log(f"    ... {msg}")

    def flush(self):
        """Log every kind's last line, then start afresh (call at the end of a step)."""
        for msg in self.pending.values():
            self._log(msg)
        self.seen, self.pending = set(), {}


@contextmanager
def step(name, summary, progress):
    """Run one step of a round, then log its name, the headline it filled in (summary["text"]) and its duration,
    and any detail lines (summary["lines"]) below."""
    t0 = time.time()
    yield
    progress.flush()
    log(f"  {name}: {summary.get('text', '')} ({time.time() - t0:.1f}s)")
    for line in summary.get("lines", []):
        log(f"      {line}")


def fmt(values, digits=4, most=None):
    """Numbers joined by spaces; past `most` of them, the rest are elided. None prints as -."""
    shown = values if most is None else values[:most]
    text = " ".join("-" if v is None else f"{v:.{digits}f}" for v in shown)
    return text + (" ..." if most is not None and len(values) > most else "")


@torch.no_grad()
def answer_table(copies, fold_of, X, Xv):
    """Every question's answer distribution on every training image through its held-out copy [N, S, V], and on
    every val image through the final copy [Nv, S, V]."""
    for c in copies:
        c.eval()
    out = torch.empty(len(X), copies[0].n_nets, copies[0].V, device=X.device)
    for f in range(len(copies) - 1):
        rows = (fold_of == f).nonzero().squeeze(1)
        for i in range(0, len(rows), FORWARD_CHUNK):
            r = rows[i:i + FORWARD_CHUNK]
            out[r] = copies[f](X[r]).softmax(-1)
    val = torch.cat([copies[-1](Xv[i:i + FORWARD_CHUNK]).softmax(-1) for i in range(0, len(Xv), FORWARD_CHUNK)])
    return out, val


def stream_batch(rows, n):
    """Equal-length index batches, one per stream, from per-stream index tensors [<= n] (short ones padded with
    their first index, or 0) -> (indices [streams, n], real [streams, n])."""
    real = torch.stack([torch.arange(n, device=r.device) < len(r) for r in rows])
    pad = [torch.cat([r, r.new_full((n - len(r),), int(r[0]) if len(r) else 0)]) for r in rows]
    return torch.stack(pad), real


@torch.no_grad()
def stream_answers(nets, images, rows):
    """Answer distributions of fused question networks (one stream per copy, nets [streams * g]) on each stream's
    own images (rows: one index tensor into images per stream) -> one [len(rows[s]), g, V] per stream."""
    out: list[list[torch.Tensor]] = [[] for _ in rows]
    for i in range(0, max(len(r) for r in rows), FORWARD_CHUNK):
        part = [r[i:i + FORWARD_CHUNK] for r in rows]
        idx, _ = stream_batch(part, max(len(r) for r in part))
        probs = nets(images[idx], len(rows)).softmax(-1)  # [streams, n, g, V]
        for s, r in enumerate(part):
            out[s].append(probs[s, :len(r)])
    return [torch.cat(o) if o else images.new_zeros(0, nets.n_nets // len(rows), nets.V) for o in out]


@torch.no_grad()
def extend(codes: CodeSet, selector, commits, gen):
    """One more answer for every sampled code: to the committed question or the selector's pick."""
    h = selector_pick(selector, commits, codes.answers, codes.questions)
    probs = codes.table[codes.images, h]
    answers = torch.multinomial(probs.cpu(), 1, generator=gen).squeeze(1).to(probs.device)
    codes.add_depth(h, answers)


def code_text(code):
    return "\u00b7".join(map(str, code)) or "root"


def node_weights(commits, table, code, cutoff):
    """Posterior weights [N] of a node whose ancestors all have committed questions (a committed or frontier node)."""
    w = torch.ones(table.shape[0], device=table.device)
    for u in range(len(code)):
        w = w * table[:, commits[code[:u]], code[u]]
        w = torch.where(w >= cutoff, w, 0.0)
    return w


@torch.no_grad()
def frontier_nodes(commits, selector, table, x, n_px, rho, cutoff, V):
    """The uncommitted children of committed nodes (just the root before any commit), heaviest first, each scored
    by the loss its children would still hold: rho^depth * images * (spread - the gain of the question the selector
    asks there now) -> [(score, code, images, spread, spread left after the current question)]."""
    if () not in commits:
        w = torch.ones(len(x), device=x.device)
        groups = [((), w[None], [()])]
    else:
        groups = []
        for z, h in commits.items():
            kids = [z + (c,) for c in range(V) if z + (c,) not in commits]
            if kids:
                w = node_weights(commits, table, z, cutoff)[None] * table[:, h, [k[-1] for k in kids]].T
                groups.append((z, torch.where(w >= cutoff, w, 0.0), kids))
    sq, S, dev = x.pow(2).sum(1), table.shape[1], x.device
    out = []
    for _, w, kids in groups:  # w: one row of posterior weights per child
        W, W2 = w.sum(1), w.pow(2).sum(1)
        D = spread(W, (w @ x).pow(2).sum(1), w @ sq, W2, n_px)
        ok = (w > 0).sum(1) >= 2
        for i, k in enumerate(kids):
            if not ok[i]:
                out.append((0.0, k, float(W[i]), 0.0, 0.0))
                continue
            left = float(D[i])
            path = [commits[k[:u]] for u in range(len(k))]  # a frontier node's ancestors are all committed
            if selector is not None and len(path) < S:
                prefix = torch.tensor([list(k)], dtype=torch.long, device=dev).view(1, len(k))
                used = torch.tensor([path], dtype=torch.long, device=dev).view(1, len(path))
                h = selector_pick(selector, commits, prefix, used)
                img = w[i].nonzero().squeeze(1)
                left -= float(posterior_gains(x, sq, table, img, w[i, img], h, n_px, cutoff, W[i], D[i])[0])
            out.append((rho ** len(k) * float(W[i]) * left, k, float(W[i]), float(D[i]), left))
    return sorted(out, key=lambda f: f[0], reverse=True)


def fit_copies(modules, batches, loss_fn, score, epochs, patience, lr, weight_decay, progress, kind, describe):
    """Train one module per copy in lockstep epochs, keeping every copy's weights from the epoch with the best
    held-out score (higher is better). batches(m) yields copy m's minibatches; loss_fn(m, batch) is its loss;
    describe(epoch, score) is the progress line, of the given kind. -> (best score, best epoch)."""
    opts = [torch.optim.AdamW([p for p in mod.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay,
                              foreach=True) for mod in modules]
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
        progress(kind, describe(epoch, s))
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
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--round-patience", type=int, default=3, help="stop after this many rounds without a better val loss")
    p.add_argument("--questions-per-round", type=int, default=8,
                   help="B: frontier nodes targeted per round, one new question each")
    p.add_argument("--codes", type=int, default=512, help="C: sampled training codes the selector is fit on")
    p.add_argument("--weight-cutoff", type=float, default=1e-4, help="posterior weights below this are dropped")
    p.add_argument("--label-weight", type=float, default=0.0,
                   help="lambda: the distance between two images also counts this much when their digits differ")
    p.add_argument("--k-decay", type=float, default=0.8, help="rho: each further answer is read with this probability")
    p.add_argument("--restarts", type=int, default=8, help="k-means++ starts per target; the best split is kept")
    p.add_argument("--hartigan-rounds", type=int, default=100, help="cap on batched Hartigan move rounds per start")
    p.add_argument("--hartigan-tol", type=float, default=1e-4,
                   help="stop Hartigan when J rises by less than this fraction per round, over the last 5 rounds")
    p.add_argument("--question-width", type=int, default=32, help="channels of each question network")
    p.add_argument("--question-hidden", type=int, default=64, help="hidden units of each question network")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=20, help="max question-network epochs per round")
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--selector-width", type=int, default=128)
    p.add_argument("--selector-layers", type=int, default=2)
    p.add_argument("--selector-heads", type=int, default=4)
    p.add_argument("--selector-batch-size", type=int, default=256)
    p.add_argument("--selector-lr", type=float, default=1e-3)
    p.add_argument("--selector-epochs", type=int, default=100, help="max selector epochs per depth")
    p.add_argument("--selector-patience", type=int, default=5)
    p.add_argument("--log-interval", type=float, default=10.0, help="seconds between progress lines")
    p.add_argument("--limit-train", type=int, default=None, help="use only the first N training images (smoke tests)")
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--out", default="checkpoints/encoder.pt")
    p.add_argument("--state", default="checkpoints/encoder_build.pt")
    p.add_argument("--plot", default="outputs/encoder_build.png", help="progress plot, redrawn after every round")
    p.add_argument("--resume", action="store_true", help="continue from --state with the next round")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    X = to_float(load_mnist("train", args.data_root, device)[: args.limit_train])
    Xv = to_float(load_mnist("val", args.data_root, device)[: args.limit_val])
    labels = load_labels("train", args.data_root, device)[: args.limit_train]
    val_labels = load_labels("val", args.data_root, device)[: args.limit_val]
    onehot, val_onehot = F.one_hot(labels, 10).float(), F.one_hot(val_labels, 10).float()
    # The objective's vectors: centered pixels and, with --label-weight, the scaled one-hot digit after them.
    # Two different one-hots are sqrt(2) apart, so the squared distance gains lambda * n_px exactly when the
    # digits differ, and every distance below (divided by n_px) gains lambda.
    feats, val_feats = X.flatten(1), Xv.flatten(1)
    if args.label_weight > 0:
        scale = math.sqrt(args.label_weight * NUM_PIXELS / 2)
        feats, val_feats = torch.cat([feats, scale * onehot], 1), torch.cat([val_feats, scale * val_onehot], 1)
    mean = feats.mean(0)
    Xc, Xvc = feats - mean, val_feats - mean  # centered, for the objective's precision
    N, Nv, V, Fo, B, C = len(X), len(Xv), args.vocab_size, args.folds, args.questions_per_round, args.codes
    P, rho, cutoff = NUM_PIXELS, args.k_decay, args.weight_cutoff
    gen = torch.Generator().manual_seed(args.seed)
    progress = Progress(args.log_interval)

    def new_selector(n_questions):
        return SelectorNet(V, n_questions, args.selector_width, args.selector_layers, args.selector_heads).to(device)

    def new_nets(n):
        return QuestionNets(n, V, args.question_width, args.question_hidden).to(device)

    state: dict[str, Any] = {"args": vars(args), "history": []}
    copies = [new_nets(0) for _ in range(Fo + 1)]  # each fold copy's bank of question networks
    selector: SelectorNet | None = None
    commits: dict[tuple[int, ...], int] = {}  # node -> the question built for it, which it always asks
    T, first_round = 0, 1
    if args.resume and os.path.exists(args.state):
        state = torch.load(args.state, map_location="cpu", weights_only=False)
        fold_of = state["fold_of"].to(device)
        copies = [new_nets(state["n_questions"]) for _ in range(Fo + 1)]
        for c, st in zip(copies, state["copies"]):
            c.load_state_dict(st)
        selector = new_selector(state["n_questions"])
        selector.load_state_dict(state["selector"])
        commits = {tuple(code): h for code, h in state["commits"]}
        T, first_round = state["T"], len(state["history"]) + 1
    else:
        if args.resume:
            log(f"--resume: no build state at {args.state}, starting from round 1")
        fold_of = (torch.randperm(N, generator=gen) % Fo).to(device)
        state["fold_of"] = fold_of.cpu()
    for c in copies:
        c.requires_grad_(False)
    table, val_table = answer_table(copies, fold_of, X, Xv)
    held_out = [(fold_of == f).nonzero().squeeze(1) for f in range(Fo)]
    Xall = torch.cat([X, Xv])  # the final copy's stream reads val images, at N and up
    log(f"device {device}, {N} training images in {Fo} folds, {Nv} val images, V={V}, {B} targets per round, "
        f"{C} codes for the selector, rho={rho}" + (f", label weight {args.label_weight}" if args.label_weight else "")
        + (f"; resuming after round {first_round - 1}" if first_round > 1 else ""))

    best_val, bad_rounds = min([h["val_loss"] for h in state["history"]], default=math.inf), 0
    try:
        for rnd in range(first_round, args.rounds + 1):
            S = copies[0].n_nets
            log(f"round {rnd}: {S} questions in the bank, codes reach depth {T}")
            entry: dict[str, Any] = {"round": rnd}

            # A. the frontier: children of committed nodes (or the root, at first), scored by the loss they hold
            summary: dict[str, Any] = {}
            with step("A frontier", summary, progress), torch.no_grad():
                frontier = frontier_nodes(commits, selector, table, Xc, P, rho, cutoff, V)
                aims = [f for f in frontier if f[0] > 0][:B]
                summary["text"] = f"{len(commits)} committed nodes, {len(frontier)} on the frontier"
                summary["lines"] = [f"target {code_text(z)}: {W:.0f} images, spread {D:.4f}, {left:.4f} after its "
                                    "current question" for _, z, W, D, left in aims]
            entry["targets"] = [(list(z), W, D, left) for _, z, W, D, left in aims]
            if not aims:
                log("stopping: no frontier node can be split")
                break

            # B. one partition per target: the best V-means split of its posterior, extended to every image by
            # nearest class centroid
            summary = {}
            with step("B partitions", summary, progress), torch.no_grad():
                Bk = len(aims)
                labels = torch.empty(Bk, N, dtype=torch.long, device=device)
                weights = torch.zeros(N, Bk, device=device)  # each image's posterior weight in each target
                implied, rounds = [], []
                for b, (_, z, _, _, _) in enumerate(aims):
                    w = node_weights(commits, table, z, cutoff)
                    img = w.nonzero().squeeze(1)
                    search = PartitionSearch(CodeSet.of_posterior(Xc, img, w[img], P, cutoff),
                                             torch.ones(1, device=device), V)
                    best: tuple[float, torch.Tensor | None] = (-math.inf, None)
                    for s in range(args.restarts):
                        where = f"B target {code_text(z)} ({b + 1}/{Bk}), start {s + 1}/{args.restarts}"
                        lab, J, r = search.run(kmeanspp(Xc, V, gen, w), torch.zeros(1, device=device),
                                               args.hartigan_rounds, args.hartigan_tol,
                                               progress=lambda r, J, where=where:
                                               progress("B", f"{where}, Hartigan round {r}: gain {J:.5f}"))
                        rounds.append(r)
                        if J > best[0]:
                            best = (J, lab)
                    J, lab = best
                    assert lab is not None
                    labels[b], weights[:, b] = search.centroid_labels(lab), w
                    implied.append(J)
                summary["text"] = f"{sum(rounds) / len(rounds):.1f} Hartigan rounds per start"
                summary["lines"] = [f"target {code_text(z)}: implied gain {J:.5f}"
                                    for (_, z, _, _, _), J in zip(aims, implied)]
            entry["implied_gain"] = implied

            # C. one new question network per target learns its partition, weighted by the target's posterior. The
            # F+1 fold copies of the Bk networks run as one fused stack, each copy reading its own images. Earlier
            # networks stay frozen and are never run again: their answers are in the table. Then each target
            # commits to its new question.
            summary = {}
            with step("C questions", summary, progress):
                new = new_nets((Fo + 1) * Bk)
                labeled = (weights.sum(1) > 0).nonzero().squeeze(1)
                # each copy's training rows and each fold's held-out rows, fixed for the round
                train_rows = [labeled[fold_of[labeled] != m] for m in range(Fo)] + [labeled]
                held_rows = [labeled[fold_of[labeled] == f] for f in range(Fo)] + [labeled[:0]]

                def ce(idx, real):
                    """Each copy's weighted cross-entropy on its batch (idx, real: [Fo + 1, n]) -> two [Fo + 1]: the
                    weighted loss sum and the weight sum."""
                    logits = new(X[idx], Fo + 1)  # [Fo + 1, n, Bk, V]
                    loss = F.cross_entropy(logits.reshape(-1, V), labels[:, idx].permute(1, 2, 0).reshape(-1),
                                           reduction="none").view(Fo + 1, -1)
                    w = (weights[idx] * real[..., None]).reshape(Fo + 1, -1)
                    return (loss * w).sum(1), w.sum(1)

                def fused_loss(_, batch):
                    total, count = ce(*batch)
                    return (total / count.clamp_min(1e-12)).sum()  # the copies' gradients don't mix

                # Every step gives each copy a full batch of its own rows, drawn from an endless shuffle; an epoch
                # is as many steps as the final copy (every labeled image) needs to see each row once.
                queues = [r[:0] for r in train_rows]

                def draw(m):
                    while len(queues[m]) < args.batch_size:
                        rows = train_rows[m]
                        queues[m] = torch.cat([queues[m], rows[torch.randperm(len(rows), generator=gen).to(device)]])
                    out, queues[m] = queues[m][:args.batch_size], queues[m][args.batch_size:]
                    return out

                def fused_batches(_):
                    for _ in range(math.ceil(len(labeled) / args.batch_size)):
                        yield stream_batch([draw(m) for m in range(Fo + 1)], args.batch_size)

                def held_out_ce():
                    total, count = torch.zeros(Fo + 1, device=device), torch.zeros(Fo + 1, device=device)
                    for i in range(0, max(len(r) for r in held_rows), FORWARD_CHUNK):
                        part = [r[i:i + FORWARD_CHUNK] for r in held_rows]
                        t, c = ce(*stream_batch(part, max(len(r) for r in part)))
                        total, count = total + t, count + c
                    return -float(total.sum()) / max(float(count.sum()), 1e-12)  # pooled over the folds

                score, epoch = fit_copies([new], fused_batches, fused_loss, held_out_ce, args.epochs, args.patience,
                                          args.lr, args.weight_decay, progress, "C",
                                          lambda e, s: f"C epoch {e}: held-out cross-entropy {-s:.4f}")
                new.requires_grad_(False)
                answers = stream_answers(new, Xall, held_out + [torch.arange(N, N + Nv, device=device)])
                fresh = torch.empty(N, Bk, V, device=device)
                for f in range(Fo):
                    fresh[held_out[f]] = answers[f]
                table, val_table = torch.cat([table, fresh], 1), torch.cat([val_table, answers[Fo]], 1)
                for c, part in zip(copies, new.split(Fo + 1)):
                    part.requires_grad_(False)
                    c.append(part)
                realized, bank_best = [], []
                sq = Xc.pow(2).sum(1)
                for b, (_, z, _, _, _) in enumerate(aims):
                    w = weights[:, b]
                    img = w.nonzero().squeeze(1)
                    used = {commits[z[:u]] for u in range(len(z))}
                    others = torch.tensor([h for h in range(S) if h not in used], dtype=torch.long, device=device)
                    asked = torch.cat([others, torch.tensor([S + b], device=device)])
                    g = posterior_gains(Xc, sq, table, img, w[img], asked, P, cutoff)
                    realized.append(float(g[-1]))
                    bank_best.append(float(g[:-1].max()) if len(others) else 0.0)
                    commits[z] = S + b
                S += Bk
                summary["text"] = f"best epoch {epoch}, held-out cross-entropy {-score:.4f}"
                summary["lines"] = [f"target {code_text(z)}: realized gain {g:.5f}, best older question there {o:.5f}"
                                    for (_, z, _, _, _), g, o in zip(aims, realized, bank_best)]
            entry.update(question_epochs=epoch, held_out_cross_entropy=-score, realized_gain=realized,
                         bank_best_gain=bank_best)

            # D. a fresh selector learns, depth by depth, which question gains most after each prefix; committed
            # prefixes always ask their own question, so only the others are its choices
            summary = {}
            with step("D selector", summary, progress):
                sels = [new_selector(S) for _ in range(Fo + 1)]
                codes = CodeSet(table, Xc, cutoff, torch.randperm(N, generator=gen)[:C].to(device), P)
                sample_fold = fold_of[codes.images]
                # per depth: every sample's gain per question [C, S], asked ones included (asking one again gains
                # about nothing, and the selector must know that if its choices above ever change)
                targets: list[torch.Tensor] = []
                scaled: list[torch.Tensor] = []  # per depth: targets / scale, what the selector regresses on
                masks: list[torch.Tensor] = []  # per depth: True for questions already asked [C, S], for picking only
                free: list[torch.Tensor] = []  # per depth: True for samples whose prefix has no committed question [C]
                pick_rows: list[list[torch.Tensor]] = []  # per depth, per fold: the free samples [n]
                fit_rows = [torch.arange(C)[sample_fold.cpu() != m] for m in range(Fo)] + [torch.arange(C)]
                drift: list[float] = []
                scale, choice = 1.0, 0.0
                for u in range(S + 1):
                    if u >= 1:
                        extend(codes, sels[Fo], commits, gen)
                    here = codes.codes_at(u)
                    if u == S or bool((codes.support[here] < 2).all()):
                        T = u
                        break
                    with torch.no_grad():
                        G = codes.question_gains(here, torch.arange(S, device=device), lambda k, n: progress(
                            "D gains", f"D gains at depth {u}: prefix {k}/{n}"))
                        local = codes.ids[:, u] - here[0]
                        targets.append(G[local])
                        masks.append(codes.used_mask(codes.ids[:, u], S))
                        free.append(torch.tensor([tuple(r) not in commits for r in codes.answers.tolist()],
                                                 dtype=torch.bool, device=device))
                        pick_rows.append([((sample_fold == f) & free[u]).nonzero().squeeze(1) for f in range(Fo)])
                    if u == 0:
                        scale = float(targets[0].abs().mean()) or 1.0
                    scaled.append(targets[u] / scale)
                    depths = list(range(u + 1))

                    def sel_batches(m, depths=depths):
                        # orders drawn on the CPU and sent over in one copy; one batch list per depth
                        rows = fit_rows[m]
                        moved = torch.cat([rows[torch.randperm(len(rows), generator=gen)] for _ in depths]).to(device)
                        out = [(d, idx) for i, d in enumerate(depths)
                               for idx in moved[i * len(rows):(i + 1) * len(rows)].split(args.selector_batch_size)]
                        return [out[i] for i in torch.randperm(len(out), generator=gen).tolist()]

                    def mse(m, batch):
                        d, idx = batch
                        pred = sels[m](codes.answers[idx, :d], codes.questions[idx, :d])
                        return (pred - scaled[d][idx]).pow(2).mean()

                    def choices(depths=depths):
                        sums = []
                        for d in depths:
                            got = torch.zeros(C, device=device)
                            for f, rows in enumerate(pick_rows[d]):
                                pred = sels[f](codes.answers[rows, :d], codes.questions[rows, :d])
                                pick = pred.masked_fill(masks[d][rows], -torch.inf).argmax(1)
                                got[rows] = targets[d][rows, pick]
                            sums.append(got.sum())
                        # committed prefixes add nothing to choose; one sync per evaluation
                        return sum(rho**d * s / C for d, s in zip(depths, torch.stack(sums).tolist()))

                    choice, _ = fit_copies(sels, sel_batches, mse, choices, args.selector_epochs, args.selector_patience,
                                           args.selector_lr, 0.0, progress, "D fit",
                                           lambda e, s, u=u: f"D depth {u}, epoch {e}: held-out gain of choices {s:.5f}")
                    for s in sels[:Fo]:
                        s.load_state_dict(sels[Fo].state_dict())
                    with torch.no_grad():
                        if u >= 1:
                            path = question_path(sels[Fo], commits, codes.answers[:, :u])
                            drift = (path != codes.questions[:, :u]).float().mean(0).tolist()
                selector = sels[Fo]
                used = codes.questions.unique().numel() if T else 0
                # at uncommitted prefixes: how often the final selector picks the question that really gains most,
                # and how much of that question's gain its picks get (near-ties cost little); None where every prefix of
                # that depth is committed
                agree: list[float | None] = []
                captured: list[float | None] = []
                with torch.no_grad():
                    for d in range(len(targets)):
                        rows = free[d].nonzero().squeeze(1)
                        if len(rows) == 0:
                            agree.append(None)
                            captured.append(None)
                            continue
                        g = targets[d][rows].masked_fill(masks[d][rows], -torch.inf)
                        pred = sels[Fo](codes.answers[rows, :d], codes.questions[rows, :d])
                        pick = pred.masked_fill(masks[d][rows], -torch.inf).argmax(1)
                        top = g.max(1)
                        agree.append(float((pick == top.indices).float().mean()))
                        got = g.gather(1, pick[:, None]).squeeze(1)
                        captured.append(float(got.sum() / top.values.sum().clamp_min(1e-12)))
                committed_share = [1 - float(f.float().mean()) for f in free]
                summary["text"] = f"codes stop at depth {T}, held-out gain of choices {choice:.5f}"
                summary["lines"] = [f"{used} of {S} questions asked"]
                if agree:  # per depth, the first 12
                    summary["lines"] += [f"share of prefixes committed    {fmt(committed_share, 2, 12)}",
                                        f"selector picks the best one    {fmt(agree, 2, 12)}",
                                        f"share of the best gain it gets {fmt(captured, 2, 12)}"]
                    if drift:
                        summary["lines"].append(f"drift                          {fmt(drift, 2, 12)}")
            entry.update(T=T, selector_choice_gain=choice, questions_asked=used, drift=drift,
                         best_question_agreement=agree, best_gain_captured=captured, committed_share=committed_share)

            # E. score on held-out images
            summary = {}
            with step("E val", summary, progress), torch.no_grad():
                assert selector is not None
                vgen = torch.Generator().manual_seed(args.seed + 7)
                vtok = torch.zeros(Nv, 0, dtype=torch.long, device=device)
                vsl = vtok.clone()
                for _ in range(T):
                    h = selector_pick(selector, commits, vtok, vsl)
                    probs = val_table[torch.arange(Nv, device=device), h]
                    t = torch.multinomial(probs.cpu(), 1, generator=vgen).to(device)
                    vtok, vsl = torch.cat([vtok, t], 1), torch.cat([vsl, h[:, None]], 1)
                sq, vsq = Xc.pow(2).sum(1), Xvc.pow(2).sum(1)
                L = torch.zeros(T + 1, device=device)
                wrong = torch.zeros(T + 1, device=device)  # posterior weight on training images of another digit
                rows_per = max(1, 2**24 // N)
                for i in range(0, Nv, rows_per):
                    v = slice(i, i + rows_per)
                    w = torch.ones(len(vsq[v]), N, device=device)
                    alive = torch.ones(len(w), dtype=torch.bool, device=device)
                    last, last_wrong = torch.zeros(len(w), device=device), torch.zeros(len(w), device=device)
                    for t in range(T + 1):
                        if t > 0:
                            w = w * table[:, vsl[v, t - 1], vtok[v, t - 1]].T
                            w = torch.where(w >= cutoff, w, 0.0)
                        W = w.sum(1)
                        dist = (W * vsq[v] - 2 * (Xvc[v] * (w @ Xc)).sum(1) + w @ sq) / (P * W.clamp_min(1e-30))
                        same = ((w @ onehot) * val_onehot[v]).sum(1) / W.clamp_min(1e-30)
                        alive &= W > 0  # past the longest prefix with weight, the loss stays where it was
                        last = torch.where(alive, dist, last)
                        last_wrong = torch.where(alive, 1 - same, last_wrong)
                        L[t] += last.sum()
                        wrong[t] += last_wrong.sum()
                L, wrong = (L / Nv).tolist(), (wrong / Nv).tolist()
                val_loss = sum((1 - rho) * rho ** (t - 1) * L[t] for t in range(1, T + 1)) + rho**T * L[T]
                summary["text"] = f"loss {val_loss:.5f}, no code {L[0]:.5f}, wrong digit {wrong[T]:.1%} at length {T}"
                summary["lines"] = [f"by length {fmt(L[1:], 5)}"]
            entry.update(val_loss=val_loss, val_loss_by_length=L, wrong_digit_by_length=wrong)
            state["history"].append(entry)

            # save the encoder (final copy and selector) and the build state
            assert selector is not None
            commit_list = [[list(z), h] for z, h in commits.items()]
            encoder = QuestionAnswerEncoder(V, S, args.question_width, args.question_hidden, args.selector_width,
                                            args.selector_layers, args.selector_heads, commit_list)
            encoder.questions.load_state_dict(copies[Fo].state_dict())
            encoder.selector.load_state_dict(selector.state_dict())
            save_checkpoint(args.out, "encoder", encoder.to(device), args=vars(args), history=state["history"])
            state.update(copies=[c.state_dict() for c in copies], n_questions=S, selector=selector.state_dict(),
                         commits=commit_list, T=T)
            torch.save(state, args.state)
            plot_build(state["history"], args.plot)

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
