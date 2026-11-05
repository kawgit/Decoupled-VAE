"""Stage 1: train the encoder, a delta tree with one network for every depth (dvae/deltatree.py's
Encoder). An image's code is the sides of the hyperplanes the network's Lloyd steps find down its path, and
a code also maps back to the network's predicted mean image at its node (the decoder, train_decoder.py, then
models the pixels around it).

Every step sends a batch of training images down the tree as the encoder now splits it. At each depth, every
node the batch reaches runs Lloyd's algorithm on the encoder from its start; the encoder is trained on the
actual side means of the batch's images there (each node's split error averaged over its Lloyd steps, each
side weighted by its image count, so every image counts once per depth); and the last step's split, detached,
takes the images to the next depth.

With --node-mean, the network also reads each node's predicted mean image (one more memory token, from a small
convolutional net), not only the history of deltas that led there.

Lloyd's starts: with --seeded, the encoder also predicts each side's top principal direction (trained on the
side's rank-1 PCA reconstruction error, weighted by --dir-weight), and a node starts from the direction its
parent predicted for it; only the root starts from the training data's top direction. With --start root, every
node starts from that same root direction. Otherwise (--start depth) every node of a depth starts from the
depth's top direction, a running average of the batches' remainders' Gram matrix there turned into its top
eigenvector every --refresh steps. With --tol > 0, a node's Lloyd steps stop once its
hyperplane moves by less than tol, and at --lloyd-steps at most; the steps taken are logged per depth.

The encoder grows one depth every --grow-every epochs (1 by default; fractions grow mid-epoch), from 1 up to
--depths (dvae/deltatree.py's Encoder.grow: new history positions and a new start), and every step trains
all its depths. The batch's split at the deepest one gives the next depth's remainders, which track that depth's
start until it joins.

With --chunk C, batches after the first epoch follow the tree: the training images, sorted by the codes the
last epoch gave them (token 0 first, ties broken at random), are cut into runs of C, the runs shuffled, and
consecutive runs joined into batches, so a batch's images share more tokens and reach fewer nodes (a node's
cost does not depend on its images). C = --batch makes each batch one run; smaller C mixes more of the tree.

Early stopping watches the held-out (val) images' mse after all --depths tokens, routed the same way, once the
encoder has grown to them, and stops a run that goes a --patience-frac share of its epochs (at least --patience)
without a better one (dvae/stopping.py); every depth's val mse is logged. A checkpoint holds everything training needs (the
optimizer's state, the tracked starts, the step and the history): saved every epoch while the encoder grows, then
at each new best. --resume continues from one, growing on to --depths if that is deeper. The progress plot
(dvae/plots.py's plot_depths) is redrawn every epoch.

Next: dump_encoder.py writes every image's code for the prior and the decoder.
"""

import argparse
import math
import os
import time

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")  # ~6.5 GB: fail instead of swapping
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")

import torch  # noqa: E402

from dvae.artifacts import artifact  # noqa: E402
from dvae.data import get_device, load_mnist  # noqa: E402
from dvae.logs import RunLog  # noqa: E402
from dvae.deltatree import (D, Encoder, Split, by_nodes, cat_splits, children, chunk_sse, lloyd,  # noqa: E402
                            node_order, pixels, root_frontier, split, top_direction)
from dvae.plots import plot_depths  # noqa: E402
from dvae.stopping import add_patience_args, stagnant  # noqa: E402


def tree_order(keys, depth):
    """Sort keys (child_key, bit j the token at depth j) for codes of depth tokens in tree order, token 0 first."""
    out = torch.zeros_like(keys)
    for j in range(depth):
        out |= ((keys >> j) & 1) << (depth - 1 - j)
    return out


def epoch_batches(N, codes, args, gen):
    """An epoch's batches of image indices [N]: random, or with --chunk and last epoch's codes [N], runs of
    --chunk images in tree order, shuffled, joined into batches."""
    if not args.chunk or codes is None:
        perm = torch.randperm(N, generator=gen)
    else:
        jitter = torch.randperm(N, generator=gen)  # ties broken at random
        order = jitter[torch.argsort(tree_order(codes, min(args.depths or 62, 62))[jitter], stable=True)]
        runs = list(order.split(args.chunk))
        perm = torch.cat([runs[i] for i in torch.randperm(len(runs), generator=gen).tolist()])
    return list(perm.split(args.batch))


def fold(grams, depth, R, ema):
    grams[depth].mul_(ema).add_(R.T @ R / len(R), alpha=1 - ema)


def depth_step(encoder, fr, args, steps):
    """One depth of train_step, steps Lloyd steps at its nodes, in pieces of at most --max-nodes nodes: the
    objective and PCA error summed over fr's images (backward done) and every node's Split."""
    if len(fr.bits) <= args.max_nodes:  # one piece: no sorting or cutting, which would wait on the GPU
        sse, pca, _, _, s = chunk_sse(encoder, fr, None, steps, tol=args.tol, dir_weight=args.dir_weight)
        (sse / (args.batch * D)).backward()
        return sse.detach(), pca, s
    order, _ = node_order(fr)
    total, pca_total = fr.R.new_zeros(()), fr.R.new_zeros(())
    ids, parts = [], []
    for piece in by_nodes(order, fr.node[order], args.max_nodes):
        sse, pca, _, piece_nodes, s = chunk_sse(encoder, fr, piece, steps, tol=args.tol, dir_weight=args.dir_weight)
        (sse / (args.batch * D)).backward()  # a fixed divisor, so every image weighs the same
        total, pca_total = total + sse.detach(), pca_total + pca
        ids.append(piece_nodes)
        parts.append(s)
    s = cat_splits(parts)
    whole = torch.argsort(torch.cat(ids))  # pieces hold whole nodes, in node order
    return total, pca_total, Split(s.n[whole], s.b[whole], s.r[whole], s.m[whole],
                                   None if s.u is None else s.u[whole], s.steps[whole])


def train_step(encoder, X, mean, grams, args, caps):
    """Send the batch X down every depth of the encoder as it splits them (depth d with caps[d] Lloyd steps),
    accumulating each depth's gradient;
    without seeds, fold the remainders each depth sees into its running Gram matrix (grams [n, D, D]), and the
    next depth's too while grams has one more (it tracks its start before it joins). Returns, per depth [K]: the
    objective and the PCA reconstruction error summed over the batch, and the Lloyd steps taken and nodes split;
    and each image's key (child_key) at the deepest depth it reached [b], on the CPU."""
    K = encoder.depths
    ahead = grams is not None and len(grams) > K
    fr = root_frontier(X, mean, resident=True)
    zero = X.new_zeros(())
    sses, pcas, steps, nodes = [zero] * K, [zero] * K, [zero] * K, [zero] * K
    for depth in range(K):
        R = fr.R
        if grams is not None:
            fold(grams, depth, R, args.ema)
        total, pca_total, s = depth_step(encoder, fr, args, caps[depth])
        sses[depth], pcas[depth] = total, pca_total
        steps[depth], nodes[depth] = s.steps.sum(), R.new_tensor(float(len(s.steps)))
        if depth + 1 < K or ahead:
            fr, _ = children(fr, s)
    if ahead:
        assert grams is not None
        fold(grams, K, fr.R, args.ema)
    return torch.stack([torch.stack(sses), torch.stack(pcas), torch.stack(steps), torch.stack(nodes)]), \
        fr.bits[fr.node.to(fr.bits.device)].cpu()


@torch.no_grad()
def refresh(encoder, grams, tracked):
    """Each tracked depth's start [n, D] from its running Gram matrix (a start keeps its sign from before), and
    the encoder's depths' starts from those."""
    for depth, gram in enumerate(grams):
        gram = gram.double() if gram.device.type == "cuda" else gram.cpu().double()  # MPS has no float64
        top = torch.linalg.eigh(gram)[1][:, -1].float().to(tracked.device)
        tracked[depth] = top if top @ tracked[depth] >= 0 else -top
    encoder.starts.copy_(tracked[:encoder.depths])


@torch.no_grad()
def lloyd_path(encoder, fr, steps, batch):
    """Every node's Split after steps Lloyd steps, and every image's squared error after the split each step
    gives [steps, N]."""
    device, Nn = fr.R.device, len(fr.bits)
    n, b, r, m = fr.R.new_zeros(Nn, D), fr.R.new_zeros(Nn), fr.R.new_zeros(Nn, D), fr.R.new_zeros(Nn)
    errs = fr.R.new_zeros(steps, len(fr.R))
    for i in range(0, Nn, batch):
        nodes = torch.arange(i, min(i + batch, Nn), device=device)
        mine = (fr.node >= i) & (fr.node < i + len(nodes))
        R, node = fr.R[mine], fr.node[mine] - i
        mems = encoder.memory(*fr.history(nodes, device))
        for t, (_, tn, tb, tr, tm, _) in enumerate(lloyd(encoder, mems, fr.starts(encoder, nodes), steps)):
            rare = (R * tn[node]).sum(-1) >= tb[node]
            errs[t, mine] = (R - torch.where(rare[:, None], tr[node], -tm[node, None] * tr[node])).square().sum(-1)
            if t == steps - 1:
                n[nodes], b[nodes], r[nodes], m[nodes] = tn, tb, tr, tm
    return Split(n, b, r, m, None, fr.R.new_full((Nn,), float(steps))), errs


@torch.no_grad()
def evaluate(encoder, X, mean, args, caps):
    """Per token of the encoder's depths [K]: the mse per pixel after it, X routed down the encoder's tree
    (depth d with caps[d] Lloyd steps); the mean Lloyd steps per node; the share of nodes that took all their
    steps; and each step's gain, the mse per pixel it takes off the step before's split (the first's off no
    split; none with --tol)."""
    encoder.eval()
    fr, mses, steps, capped, gains = root_frontier(X, mean, resident=True), [], [], [], []
    for depth in range(encoder.depths):
        if args.tol > 0:
            fr, _, s = split(encoder, fr, caps[depth], args.node_batch, args.tol)
            gains.append([])
        else:
            s, errs = lloyd_path(encoder, fr, caps[depth], args.node_batch)
            before = torch.cat([fr.R.square().sum(-1)[None], errs[:-1]])
            gains.append(((before - errs).mean(1) / D).tolist())
            fr, _ = children(fr, s)
        mses.append(fr.R.square().mean().item())
        steps.append(s.steps.mean().item())
        capped.append((s.steps >= caps[depth]).float().mean().item())
    return mses, steps, capped, gains


def grow(encoder, opt, depths, starts):
    """encoder.grow, and the optimizer's state for the position rows (zeros for the new rows, what zero
    gradients would have left them)."""
    p = encoder.position
    old = len(p)
    encoder.grow(depths, starts)
    for key, value in opt.state.get(p, {}).items():
        if torch.is_tensor(value) and value.shape[:1] == (old,):
            opt.state[p][key] = torch.cat([value, value.new_zeros(len(p) - old, *value.shape[1:])])


def psnr(mse):
    return -10 * math.log10(mse)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depths", type=int, default=10,
                   help="tokens per code, where the encoder stops growing (0: no cap; then nothing stops the run but "
                        "--epochs, as early stopping waits for the full depth)")
    p.add_argument("--grow-every", type=float, default=1.0,
                   help="epochs between depths added (fractions allowed: 0.5 adds two per epoch)")
    p.add_argument("--resume", default=None, metavar="CKPT",
                   help="continue from a checkpoint (its architecture and data mean; training settings from now)")
    p.add_argument("--lloyd-steps", type=int, default=4,
                   help="Lloyd steps per node (with --tol, the most a node takes)")
    p.add_argument("--tol", type=float, default=0.0, help="stop a node's Lloyd steps once its hyperplane moves less")
    p.add_argument("--seeded", action="store_true", help="start each node from its parent's predicted direction")
    p.add_argument("--node-mean", action="store_true",
                   help="also show the network each node's predicted mean image (dvae/deltatree.py's HistoryReader)")
    p.add_argument("--start", choices=("depth", "root"), default="depth",
                   help="without --seeded, each node's Lloyd start: its depth's tracked top direction, or the root's")
    p.add_argument("--dir-weight", type=float, default=1.0, help="weight of the seeds' PCA reconstruction error")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--batch", type=int, default=1024, help="training images per step")
    p.add_argument("--chunk", type=int, default=0, help="batch runs of this many images in last epoch's code order")
    p.add_argument("--tf32", action="store_true", help="TF32 matmuls on GPUs that have them (~20%% faster on an L40S)")
    p.add_argument("--max-nodes", type=int, default=256,
                   help="nodes per forward pass; a depth spanning more is run in pieces with accumulated gradients")
    p.add_argument("--node-batch", type=int, default=512, help="nodes split at once when evaluating")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--ema", type=float, default=0.95, help="running statistics' decay per step (without --seeded)")
    p.add_argument("--refresh", type=int, default=50, help="steps between start updates (without --seeded)")
    p.add_argument("--epochs", type=int, default=200)
    add_patience_args(p)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--out", default=None, help="checkpoint (default: model.pt in the run's artifacts folder)")
    p.add_argument("--log", default=None, help="summary log (default: run.log in the run's artifacts folder)")
    p.add_argument("--plot", default=None, help="progress plot, redrawn every epoch (default: --log's path as .png)")
    p.add_argument("--log-interval", type=float, default=30.0, help="seconds between step logs")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="data")
    args = p.parse_args()
    args.out, args.log = args.out or artifact("model.pt"), args.log or artifact("run.log")

    device = get_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    log = RunLog(args.log, device)

    ckpt = torch.load(args.resume, map_location=device, weights_only=False) if args.resume else None
    if ckpt is not None:  # the architecture is the checkpoint's
        cfg = ckpt["shared"]["config"]
        args.width, args.layers, args.heads, args.seeded = cfg["width"], cfg["layers"], cfg["heads"], cfg["seeded"]
        args.node_mean = cfg.get("node_mean", False)
    log(f"=== {time.strftime('%Y-%m-%d')} {vars(args)}")
    X = pixels(load_mnist("train", args.data_root)[:args.limit_train]).to(device)
    V = pixels(load_mnist("val", args.data_root)[:args.limit_val]).to(device)
    mean = X.mean(0) if ckpt is None else ckpt["mean"].to(device)
    K, N = args.depths if args.depths > 0 else math.inf, len(X)
    R = X - mean
    root_top = top_direction(R)
    root_gram = R.T @ R / N
    tracking = not args.seeded and args.start == "depth"

    torch.manual_seed(args.seed)
    if ckpt is None:
        encoder = Encoder(1, args.width, args.layers, args.heads, seeded=args.seeded,
                          node_mean=args.node_mean).to(device)
        with torch.no_grad():
            encoder.starts[0] = root_top
            if args.node_mean:
                encoder.mean.copy_(mean)
    else:
        encoder = Encoder(**ckpt["shared"]["config"]).to(device)
        encoder.load_state_dict(ckpt["shared"]["state_dict"])
        assert encoder.depths <= K, f"--depths {K} is below the checkpoint's {encoder.depths}"
    # each depth's tracked start, the encoder's and the next one's (the root's direction until tracked)
    tracked = torch.cat([encoder.starts.detach().clone(), root_top[None]])[:min(encoder.depths + 1, K)]
    grams = None
    if tracking:
        grams = root_gram.expand(len(tracked), D, D).clone()
        if ckpt is not None and ckpt.get("grams") is not None:
            kept = ckpt["grams"].to(device)[:len(tracked)]
            grams[:len(kept)] = kept
            tracked[:len(kept)] = ckpt["tracked"].to(device)[:len(kept)]
    log(f"shared encoder: {sum(q.numel() for q in encoder.parameters()) / 1e6:.1f}M parameters, "
        f"{encoder.depths} depths, growing to {K}")

    opt = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history, step, first, best = [], 0, 0, (math.inf, -1)
    grown = 0.0  # when (in epochs) the encoder last grew
    caps = [args.lloyd_steps] * encoder.depths  # each depth's Lloyd steps, part of the tree
    if ckpt is not None:
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
            for group in opt.param_groups:  # the settings given now
                group.update(lr=args.lr, initial_lr=args.lr, weight_decay=args.weight_decay)
        else:
            log("the checkpoint has no optimizer state: AdamW starts afresh")
        history, step, first = ckpt["metrics"], ckpt.get("step", 0), ckpt["epoch"] + 1
        grown = max(ckpt.get("grown", -math.inf), first - args.grow_every)  # grow as it resumes, not to catch up
        caps = list(ckpt.get("caps") or [ckpt["args"]["lloyd_steps"]] * encoder.depths)
        if encoder.depths == K and ckpt.get("best"):
            best = tuple(ckpt["best"])  # the same full depth: early stopping carries on
        log(f"resumed {args.resume} after epoch {ckpt['epoch']} (step {step}), saving to {args.out}")
    step0 = step
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (step0 + s + 1) / args.warmup_steps))
    began = time.time() - (history[-1].get("seconds", 0.0) if history else 0.0)
    codes = None  # each training image's code in the last epoch

    def save(epoch):
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save(dict(args=vars(args), mean=mean.cpu(), epoch=epoch, step=step, metrics=history, best=best,
                        shared=dict(config=encoder.config,
                                    state_dict={k: v.cpu() for k, v in encoder.state_dict().items()}),
                        optimizer=opt.state_dict(), tracked=tracked.cpu(), grown=grown, caps=caps,
                        grams=None if grams is None else grams.cpu()), args.out)

    def fmt(values, digits=5):
        return " ".join(f"{v:.{digits}f}" for v in values)

    def add_depth():
        """One more depth, from its start tracked so far (refreshed now), and track the next one's."""
        nonlocal tracked, grams
        depth = encoder.depths
        if grams is not None:
            refresh(encoder, grams, tracked)
        grow(encoder, opt, depth + 1, tracked[depth:depth + 1])
        caps.append(args.lloyd_steps)
        if depth + 1 < K:
            tracked = torch.cat([tracked, root_top[None]])
            if grams is not None:
                grams = torch.cat([grams, root_gram[None]])

    for epoch in range(first, first + args.epochs):
        t0, last = time.time(), time.time()
        encoder.train()
        totals = torch.zeros(4, encoder.depths, device=device)  # (widened as the encoder grows)
        gen = torch.Generator().manual_seed(args.seed * 1_000_003 + epoch)
        batches, seen = epoch_batches(N, codes, args, gen), 0
        codes = torch.zeros(N, dtype=torch.int64)
        for i, idx in enumerate(batches):
            if encoder.depths < K and epoch + i / len(batches) - grown >= args.grow_every - 1e-9:
                add_depth()
                grown += args.grow_every
            opt.zero_grad(set_to_none=True)
            stats, keys = train_step(encoder, X[idx.to(device)], mean, grams, args, caps)
            if stats.shape[1] > totals.shape[1]:
                totals = torch.cat([totals, totals.new_zeros(4, stats.shape[1] - totals.shape[1])], 1)
            totals[:, :stats.shape[1]] += stats
            codes[idx] = keys
            seen += len(idx)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if grams is not None and step % args.refresh == 0:
                refresh(encoder, grams, tracked)
            if time.time() - last >= args.log_interval:
                last = time.time()
                log.detail(f"    step {step} ({seen}/{N} images this epoch)")
        t1 = time.time()
        val, val_steps, capped, step_gains = evaluate(encoder, V, mean, args, caps)
        totals = totals[:, :encoder.depths]  # (a depth added mid-epoch counts only the images it saw)
        objective = (totals[0] / (N * D)).tolist()
        pca = (totals[1] / (N * D)).tolist()
        train_steps = (totals[2] / totals[3].clamp_min(1)).tolist()
        full = encoder.depths == K
        improved = full and val[-1] < best[0]
        if improved:
            best = (val[-1], epoch)
        history.append(dict(epoch=epoch, val_mse=val, objective=objective, pca=pca, train_steps=train_steps,
                            val_steps=val_steps, val_capped=capped, trained=encoder.depths - 1, caps=list(caps),
                            step_gains=step_gains, seconds=time.time() - began))
        plot_depths(history, args.plot or os.path.splitext(args.log)[0] + ".png")
        log(f"  epoch {epoch:2d}: val mse after {encoder.depths} tokens {val[-1]:.6f} (psnr {psnr(val[-1]):.2f} dB); "
            "per depth " + fmt(val) + "; objective per depth " + fmt(objective)
            + (("; pca per depth " + fmt(pca)) if args.seeded else "")
            + "; lloyd steps per depth train " + fmt(train_steps, 2) + ", val " + fmt(val_steps, 2)
            + ", val capped " + fmt(capped, 2) + "; nodes per batch " + fmt((totals[3] / len(batches)).tolist(), 0)
            + f" ({t1 - t0:.0f}s train, {time.time() - t1:.0f}s val){'  *' if improved else ''}")
        if not full or improved:
            save(epoch)  # the latest while growing, then the best
        elif stagnant(epoch, best[1], args):
            break
    if best[1] < 0:
        log(f"stopped at {encoder.depths} of {K} depths; the latest state is in {args.out}")
        return
    log(f"kept epoch {best[1]}: val mse after {K} tokens {best[0]:.6f} (psnr {psnr(best[0]):.2f} dB), saved {args.out}")


if __name__ == "__main__":
    main()
