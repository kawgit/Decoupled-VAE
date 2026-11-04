"""Train a delta tree in one stage, with one decoder for every depth (dvae/deltatree.py's SharedDecoder).

Every step sends a batch of training images down the tree as the decoder now splits it. At each depth, every
node the batch reaches runs Lloyd's algorithm on the decoder from its start; the decoder is trained on the
actual side means of the batch's images there (each node's split error averaged over its Lloyd steps, each
side weighted by its image count, so every image counts once per depth); and the last step's split, detached,
takes the images to the next depth.

Lloyd's starts: with --seeded, the decoder also predicts each side's top principal direction (trained on the
side's rank-1 PCA reconstruction error, weighted by --dir-weight), and a node starts from the direction its
parent predicted for it; only the root starts from the training data's top direction. With --start root, every
node starts from that same root direction. Otherwise (--start depth) every node of a depth starts from the
depth's top direction, a running average of the batches' remainders' Gram matrix there turned into its top
eigenvector every --refresh steps. With --tol > 0, a node's Lloyd steps stop once its
hyperplane moves by less than tol, and at --lloyd-steps at most; the steps taken are logged per depth.

The decoder grows one depth every --grow-every epochs (1 by default; fractions grow mid-epoch), from 1 up to
--depths (dvae/deltatree.py's SharedDecoder.grow: new history positions and a new start), and every step trains
all its depths. The batch's split at the deepest one gives the next depth's remainders, which track that depth's
start until it joins.

With --chunk C, batches after the first epoch follow the tree: the training images, sorted by the codes the
last epoch gave them (token 0 first, ties broken at random), are cut into runs of C, the runs shuffled, and
consecutive runs joined into batches, so a batch's images share more tokens and reach fewer nodes (a node's
cost does not depend on its images). C = --batch makes each batch one run; smaller C mixes more of the tree.

Early stopping watches the held-out (val) images' mse after all --depths tokens, routed the same way, once the
decoder has grown to them; every depth's val mse is logged. A checkpoint holds everything training needs (the
optimizer's state, the tracked starts, the step and the history): saved every epoch while the decoder grows, then
at each new best. --resume continues from one, growing on to --depths if that is deeper.
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
from dvae.deltatree import (D, SharedDecoder, Split, by_nodes, cat_splits, children, chunk_sse, lloyd,  # noqa: E402
                            node_order, pixels, root_frontier, split, top_direction)
from dvae.graphs import GraphedDepths  # noqa: E402
from dvae.plots import plot_depths  # noqa: E402


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
        order = jitter[torch.argsort(tree_order(codes, min(args.depths, 62))[jitter], stable=True)]
        runs = list(order.split(args.chunk))
        perm = torch.cat([runs[i] for i in torch.randperm(len(runs), generator=gen).tolist()])
    return list(perm.split(args.batch))


def fold(grams, depth, R, ema):
    grams[depth].mul_(ema).add_(R.T @ R / len(R), alpha=1 - ema)


def depth_step(decoder, fr, args, steps):
    """One depth of train_step, steps Lloyd steps at its nodes, in pieces of at most --max-nodes nodes: the
    objective and PCA error summed over fr's images (backward done) and every node's Split."""
    if len(fr.bits) <= args.max_nodes:  # one piece: no sorting or cutting, which would wait on the GPU
        sse, pca, _, _, s = chunk_sse(decoder, fr, None, steps, tol=args.tol, dir_weight=args.dir_weight)
        (sse / (args.batch * D)).backward()
        return sse.detach(), pca, s
    order, _ = node_order(fr)
    total, pca_total = fr.R.new_zeros(()), fr.R.new_zeros(())
    ids, parts = [], []
    for piece in by_nodes(order, fr.node[order], args.max_nodes):
        sse, pca, _, piece_nodes, s = chunk_sse(decoder, fr, piece, steps, tol=args.tol, dir_weight=args.dir_weight)
        (sse / (args.batch * D)).backward()  # a fixed divisor, so every image weighs the same
        total, pca_total = total + sse.detach(), pca_total + pca
        ids.append(piece_nodes)
        parts.append(s)
    s = cat_splits(parts)
    whole = torch.argsort(torch.cat(ids))  # pieces hold whole nodes, in node order
    return total, pca_total, Split(s.n[whole], s.b[whole], s.r[whole], s.m[whole],
                                   None if s.u is None else s.u[whole], s.steps[whole])


def train_step(decoder, X, mean, grams, args, caps, graphs: GraphedDepths | None = None):
    """Send the batch X down every depth of the decoder as it splits them (depth d with caps[d] Lloyd steps),
    accumulating each depth's gradient;
    without seeds, fold the remainders each depth sees into its running Gram matrix (grams [n, D, D]), and the
    next depth's too while grams has one more (it tracks its start before it joins). Returns, per depth [K]: the
    objective and the PCA reconstruction error summed over the batch, and the Lloyd steps taken and nodes split;
    and each image's key (child_key) at the deepest depth it reached [b], on the CPU. With graphs, a full
    batch's depths run as CUDA graphs."""
    K = decoder.depths
    ahead = grams is not None and len(grams) > K
    fr = root_frontier(X, mean, resident=True)
    zero = X.new_zeros(())
    sses, pcas, steps, nodes = [zero] * K, [zero] * K, [zero] * K, [zero] * K
    for depth in range(K):
        R = fr.R
        if grams is not None:
            fold(grams, depth, R, args.ema)
        if graphs is not None and len(X) == graphs.batch:
            total, s = graphs.run(fr, caps[depth])
            pca_total = zero
        else:
            total, pca_total, s = depth_step(decoder, fr, args, caps[depth])
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
def refresh(decoder, grams, tracked):
    """Each tracked depth's start [n, D] from its running Gram matrix (a start keeps its sign from before), and
    the decoder's depths' starts from those."""
    for depth, gram in enumerate(grams):
        gram = gram.double() if gram.device.type == "cuda" else gram.cpu().double()  # MPS has no float64
        top = torch.linalg.eigh(gram)[1][:, -1].float().to(tracked.device)
        tracked[depth] = top if top @ tracked[depth] >= 0 else -top
    decoder.starts.copy_(tracked[:decoder.depths])


@torch.no_grad()
def lloyd_path(decoder, fr, steps, extra, batch):
    """Every node's Split after steps Lloyd steps, and every image's squared error after the split each of the
    first steps + extra steps gives [steps + extra, N]."""
    device, Nn = fr.R.device, len(fr.bits)
    n, b, r, m = fr.R.new_zeros(Nn, D), fr.R.new_zeros(Nn), fr.R.new_zeros(Nn, D), fr.R.new_zeros(Nn)
    errs = fr.R.new_zeros(steps + extra, len(fr.R))
    for i in range(0, Nn, batch):
        nodes = torch.arange(i, min(i + batch, Nn), device=device)
        mine = (fr.node >= i) & (fr.node < i + len(nodes))
        R, node = fr.R[mine], fr.node[mine] - i
        mems = decoder.memory(*fr.history(nodes, device))
        for t, (_, tn, tb, tr, tm, _) in enumerate(lloyd(decoder, mems, fr.starts(decoder, nodes), steps + extra)):
            rare = (R * tn[node]).sum(-1) >= tb[node]
            errs[t, mine] = (R - torch.where(rare[:, None], tr[node], -tm[node, None] * tr[node])).square().sum(-1)
            if t == steps - 1:
                n[nodes], b[nodes], r[nodes], m[nodes] = tn, tb, tr, tm
    return Split(n, b, r, m, None, fr.R.new_full((Nn,), float(steps))), errs


@torch.no_grad()
def evaluate(decoder, X, mean, args, caps, test):
    """Per token of the decoder's depths [K]: the mse per pixel after it, X routed down the decoder's tree
    (depth d with caps[d] Lloyd steps); the mean Lloyd steps per node; the share of nodes that took all their
    steps; at each depth to test (test[d]) the z-score of one more step's gain (its mean over the images, in
    standard errors), else None; and each step's gain, the mse per pixel it takes off the step before's split
    (the first's off no split), through the look-ahead step at a tested depth (none with --tol)."""
    decoder.eval()
    fr, mses, steps, capped, zs, gains = root_frontier(X, mean, resident=True), [], [], [], [], []
    for depth in range(decoder.depths):
        if args.tol > 0:
            fr, _, s = split(decoder, fr, caps[depth], args.node_batch, args.tol)
            zs.append(None)
            gains.append([])
        else:
            s, errs = lloyd_path(decoder, fr, caps[depth], int(test[depth]), args.node_batch)
            before = torch.cat([fr.R.square().sum(-1)[None], errs[:-1]])
            gains.append(((before - errs).mean(1) / D).tolist())
            if test[depth]:
                gain = errs[-2] - errs[-1]
                se = gain.std() / len(gain) ** 0.5
                zs.append(gain.mean().item() / se.item() if se > 0 else (math.inf if gain.mean() > 0 else 0.0))
            else:
                zs.append(None)
            fr, _ = children(fr, s)
        mses.append(fr.R.square().mean().item())
        steps.append(s.steps.mean().item())
        capped.append((s.steps >= caps[depth]).float().mean().item())
    return mses, steps, capped, zs, gains


def grow(decoder, opt, depths, starts):
    """decoder.grow, and the optimizer's state for the position rows (zeros for the new rows, what zero
    gradients would have left them)."""
    p = decoder.position
    old = len(p)
    decoder.grow(depths, starts)
    for key, value in opt.state.get(p, {}).items():
        if torch.is_tensor(value) and value.shape[:1] == (old,):
            opt.state[p][key] = torch.cat([value, value.new_zeros(len(p) - old, *value.shape[1:])])


def psnr(mse):
    return -10 * math.log10(mse)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depths", type=int, default=10, help="tokens per code, where the decoder stops growing")
    p.add_argument("--grow-every", type=float, default=1.0,
                   help="epochs between depths added (fractions allowed: 0.5 adds two per epoch)")
    p.add_argument("--resume", default=None, metavar="CKPT",
                   help="continue from a checkpoint (its architecture and data mean; training settings from now)")
    p.add_argument("--lloyd-steps", type=int, default=4,
                   help="Lloyd steps per node (with --tol, the most a node takes); without --lloyd-curriculum")
    p.add_argument("--lloyd-curriculum", action="store_true",
                   help="each depth starts at 1 Lloyd step and, every epoch, gains one if one more step's val gain "
                        "is significant (above 2 standard errors)")
    p.add_argument("--lloyd-max", type=int, default=32, help="with --lloyd-curriculum, a ceiling no depth passes")
    p.add_argument("--tol", type=float, default=0.0, help="stop a node's Lloyd steps once its hyperplane moves less")
    p.add_argument("--seeded", action="store_true", help="start each node from its parent's predicted direction")
    p.add_argument("--start", choices=("depth", "root"), default="depth",
                   help="without --seeded, each node's Lloyd start: its depth's tracked top direction, or the root's")
    p.add_argument("--dir-weight", type=float, default=1.0, help="weight of the seeds' PCA reconstruction error")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--batch", type=int, default=1024, help="training images per step")
    p.add_argument("--chunk", type=int, default=0, help="batch runs of this many images in last epoch's code order")
    p.add_argument("--tf32", action="store_true", help="TF32 matmuls on GPUs that have them (~20%% faster on an L40S)")
    p.add_argument("--graphs", action="store_true",
                   help="run full batches' depths as CUDA graphs (CUDA only; no --seeded or --tol)")
    p.add_argument("--max-nodes", type=int, default=256,
                   help="nodes per forward pass; a depth spanning more is run in pieces with accumulated gradients")
    p.add_argument("--node-batch", type=int, default=512, help="nodes split at once when evaluating")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--ema", type=float, default=0.95, help="running statistics' decay per step (without --seeded)")
    p.add_argument("--refresh", type=int, default=50, help="steps between start updates (without --seeded)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=5)
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
    log(f"=== {time.strftime('%Y-%m-%d')} {vars(args)}")
    X = pixels(load_mnist("train", args.data_root)[:args.limit_train]).to(device)
    V = pixels(load_mnist("val", args.data_root)[:args.limit_val]).to(device)
    mean = X.mean(0) if ckpt is None else ckpt["mean"].to(device)
    K, N = args.depths, len(X)
    R = X - mean
    root_top = top_direction(R)
    root_gram = R.T @ R / N
    tracking = not args.seeded and args.start == "depth"
    assert not args.lloyd_curriculum or (not args.seeded and args.tol == 0), "--lloyd-curriculum: no --seeded or --tol"
    first_cap = 1 if args.lloyd_curriculum else args.lloyd_steps

    torch.manual_seed(args.seed)
    if ckpt is None:
        decoder = SharedDecoder(1, args.width, args.layers, args.heads, seeded=args.seeded).to(device)
        with torch.no_grad():
            decoder.starts[0] = root_top
    else:
        decoder = SharedDecoder(**ckpt["shared"]["config"]).to(device)
        decoder.load_state_dict(ckpt["shared"]["state_dict"])
        assert decoder.depths <= K, f"--depths {K} is below the checkpoint's {decoder.depths}"
    # each depth's tracked start, the decoder's and the next one's (the root's direction until tracked)
    tracked = torch.cat([decoder.starts.detach().clone(), root_top[None]])[:min(decoder.depths + 1, K)]
    grams = None
    if tracking:
        grams = root_gram.expand(len(tracked), D, D).clone()
        if ckpt is not None and ckpt.get("grams") is not None:
            kept = ckpt["grams"].to(device)[:len(tracked)]
            grams[:len(kept)] = kept
            tracked[:len(kept)] = ckpt["tracked"].to(device)[:len(kept)]
    log(f"shared decoder: {sum(q.numel() for q in decoder.parameters()) / 1e6:.1f}M parameters, "
        f"{decoder.depths} depths, growing one per epoch to {K}")

    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history, step, first, best = [], 0, 0, (math.inf, -1)
    grown = 0.0  # when (in epochs) the decoder last grew
    caps = [first_cap] * decoder.depths  # each depth's Lloyd steps, part of the tree
    if ckpt is not None:
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
            for group in opt.param_groups:  # the settings given now
                group.update(lr=args.lr, initial_lr=args.lr, weight_decay=args.weight_decay)
        else:
            log("the checkpoint has no optimizer state: AdamW starts afresh")
        history, step, first = ckpt["metrics"], ckpt.get("step", 0), ckpt["epoch"] + 1
        grown = max(ckpt.get("grown", -math.inf), first - args.grow_every)  # grow as it resumes, not to catch up
        caps = list(ckpt.get("caps") or [ckpt["args"]["lloyd_steps"]] * decoder.depths)
        if decoder.depths == K and ckpt.get("best"):
            best = tuple(ckpt["best"])  # the same full depth: early stopping carries on
        log(f"resumed {args.resume} after epoch {ckpt['epoch']} (step {step}), saving to {args.out}")
    step0 = step
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (step0 + s + 1) / args.warmup_steps))
    graphs = None
    if args.graphs:
        assert device.type == "cuda" and not args.seeded and args.tol == 0, "--graphs: CUDA, no --seeded or --tol"
    began = time.time() - (history[-1].get("seconds", 0.0) if history else 0.0)
    codes = None  # each training image's code in the last epoch

    def save(epoch):
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save(dict(args=vars(args), mean=mean.cpu(), epoch=epoch, step=step, metrics=history, best=best,
                        shared=dict(config=decoder.config,
                                    state_dict={k: v.cpu() for k, v in decoder.state_dict().items()}),
                        optimizer=opt.state_dict(), tracked=tracked.cpu(), grown=grown, caps=caps,
                        grams=None if grams is None else grams.cpu()), args.out)

    def fmt(values, digits=5):
        return " ".join(f"{v:.{digits}f}" for v in values)

    def add_depth():
        """One more depth, from its start tracked so far (refreshed now), and track the next one's."""
        nonlocal tracked, grams, graphs
        depth = decoder.depths
        if grams is not None:
            refresh(decoder, grams, tracked)
        grow(decoder, opt, depth + 1, tracked[depth:depth + 1])
        caps.append(first_cap)
        if depth + 1 < K:
            tracked = torch.cat([tracked, root_top[None]])
            if grams is not None:
                grams = torch.cat([grams, root_gram[None]])
        graphs = None  # recorded again for the grown parameters

    for epoch in range(first, first + args.epochs):
        t0, last = time.time(), time.time()
        decoder.train()
        totals = torch.zeros(4, K, device=device)
        gen = torch.Generator().manual_seed(args.seed * 1_000_003 + epoch)
        batches, seen = epoch_batches(N, codes, args, gen), 0
        codes = torch.zeros(N, dtype=torch.int64)
        for i, idx in enumerate(batches):
            if decoder.depths < K and epoch + i / len(batches) - grown >= args.grow_every - 1e-9:
                add_depth()
                grown += args.grow_every
            if args.graphs and graphs is None:  # (re)recorded for the decoder's parameters as they are now
                graphs = GraphedDepths(decoder, args.batch, args.batch * D)
            opt.zero_grad(set_to_none=graphs is None)  # graphs add into the gradients' fixed buffers
            stats, keys = train_step(decoder, X[idx.to(device)], mean, grams, args, caps, graphs)
            totals[:, :len(stats[0])] += stats
            codes[idx] = keys
            seen += len(idx)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if grams is not None and step % args.refresh == 0:
                refresh(decoder, grams, tracked)
            if time.time() - last >= args.log_interval:
                last = time.time()
                log(f"    step {step} ({seen}/{N} images this epoch)")
        t1 = time.time()
        trained_caps = list(caps)
        test = [args.lloyd_curriculum and c < args.lloyd_max for c in caps]
        val, val_steps, capped, zs, step_gains = evaluate(decoder, V, mean, args, caps, test)
        for depth, z in enumerate(zs):  # one more step wherever it would measurably help
            if z is not None and z > 2:
                caps[depth] += 1
        totals = totals[:, :decoder.depths]  # (a depth added mid-epoch counts only the images it saw)
        objective = (totals[0] / (N * D)).tolist()
        pca = (totals[1] / (N * D)).tolist()
        train_steps = (totals[2] / totals[3].clamp_min(1)).tolist()
        full = decoder.depths == K
        improved = full and val[-1] < best[0]
        if improved:
            best = (val[-1], epoch)
        history.append(dict(epoch=epoch, val_mse=val, objective=objective, pca=pca, train_steps=train_steps,
                            val_steps=val_steps, val_capped=capped, trained=decoder.depths - 1, caps=trained_caps,
                            look_ahead=zs, step_gains=step_gains,
                            seconds=time.time() - began))
        plot_depths(history, args.plot or os.path.splitext(args.log)[0] + ".png")
        log(f"  epoch {epoch:2d}: val mse after {decoder.depths} tokens {val[-1]:.6f} (psnr {psnr(val[-1]):.2f} dB); "
            "per depth " + fmt(val) + "; objective per depth " + fmt(objective)
            + (("; pca per depth " + fmt(pca)) if args.seeded else "")
            + "; lloyd steps per depth train " + fmt(train_steps, 2) + ", val " + fmt(val_steps, 2)
            + ", val capped " + fmt(capped, 2) + "; nodes per batch " + fmt((totals[3] / len(batches)).tolist(), 0)
            + ("; lloyd caps " + " ".join(f"{c}{'+' if new > c else ''}" for c, new in zip(trained_caps, caps))
               + " (+ one more next epoch), look-ahead z " + " ".join("-" if z is None else f"{z:.1f}" for z in zs)
               if args.lloyd_curriculum else "")
            + f" ({t1 - t0:.0f}s train, {time.time() - t1:.0f}s val){'  *' if improved else ''}")
        if not full or improved:
            save(epoch)  # the latest while growing, then the best
        elif epoch - best[1] >= args.patience:
            break
    if best[1] < 0:
        log(f"stopped at {decoder.depths} of {K} depths; the latest state is in {args.out}")
        return
    log(f"kept epoch {best[1]}: val mse after {K} tokens {best[0]:.6f} (psnr {psnr(best[0]):.2f} dB), saved {args.out}")


if __name__ == "__main__":
    main()
