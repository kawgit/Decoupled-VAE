"""Stage 1, two-centroid variant: train dvae/centroids.py's CentroidEncoder, which predicts each node's two
centroids directly instead of running Lloyd's algorithm through predicted side means (train_encoder.py).

Every step sends a batch of training images down the tree as the encoder now splits it. At each depth, every node
the batch reaches gets its two centroids from the network, each image goes to the nearer one, and the network is
trained on each image's squared distance to that centroid (the k-means objective, every image counted once per
depth); the split, detached, takes the images to the next depth. The gradient pulls each centroid toward the mean
of the images it won, so training does Lloyd's algorithm across batches. Each epoch logs, per depth, the share of
nodes (of those holding at least two of the batch's images) that sent every image to one centroid: a centroid
that never wins gets no gradient.

As in train_encoder.py, the encoder grows one depth every --grow-every epochs, from 1 up to --depths (0: no cap),
and every step trains all its depths; early stopping watches the val images' mse after all --depths tokens once the
encoder has grown to them, and stops a run that goes a --patience-frac share of its epochs (at least --patience)
without a better one (dvae/stopping.py). A checkpoint holds everything training needs; it is saved every epoch while the encoder
grows, then at each new best, and --resume continues from one. The progress plot (dvae/plots.py's plot_depths) is
redrawn every epoch.

The later stages (dump_encoder.py and on) still read train_encoder.py's (r, m) splits, not these checkpoints.
"""

import argparse
import math
import os
import time

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")  # ~6.5 GB: fail instead of swapping
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")

import torch  # noqa: E402

from dvae.artifacts import artifact  # noqa: E402
from dvae.centroids import CentroidEncoder, centroid_children, nearer  # noqa: E402
from dvae.data import get_device, load_mnist  # noqa: E402
from dvae.deltatree import D, Frontier, by_nodes, node_order, pixels, root_frontier  # noqa: E402
from dvae.logs import RunLog  # noqa: E402
from dvae.plots import plot_depths  # noqa: E402
from dvae.stopping import add_patience_args, stagnant  # noqa: E402


def depth_step(encoder, fr: Frontier, args):
    """One depth of train_step, in pieces of at most --max-nodes nodes: the objective summed over fr's images
    (backward done) and every node's centroids [Nn, 2, D], detached."""
    Nn, device = len(fr.bits), fr.R.device
    C = fr.R.new_zeros(Nn, 2, D)
    total = fr.R.new_zeros(())
    if Nn <= args.max_nodes:  # one piece: no sorting or cutting, which would wait on the GPU
        pieces = [None]
    else:
        order, _ = node_order(fr)
        pieces = by_nodes(order, fr.node[order], args.max_nodes)
    for imgs in pieces:
        if imgs is None:
            R, inverse, nodes = fr.R, fr.node, torch.arange(Nn, device=device)
        else:
            R = fr.R[imgs]
            nodes, inverse = torch.unique(fr.node[imgs], return_inverse=True)
        c = encoder.centroids(encoder.memory(*fr.history(nodes, device)))
        loss = nearer(R, c[inverse])[1].sum()
        (loss / (args.batch * D)).backward()  # a fixed divisor, so every image weighs the same
        total = total + loss.detach()
        C[nodes] = c.detach()
    return total, C


def one_sided(fr: Frontier, side):
    """How many nodes holding at least two of fr's images sent all of them to one centroid, and how many hold two
    or more."""
    Nn = len(fr.bits)
    counts = torch.bincount(fr.node, minlength=Nn)
    ones = torch.bincount(fr.node, weights=side.float(), minlength=Nn)
    held = counts >= 2
    return (held & ((ones == 0) | (ones == counts))).sum(), held.sum()


def train_step(encoder, X, mean, args):
    """Send the batch X down every depth of the encoder as it splits them, accumulating each depth's gradient.
    Returns, per depth [K]: the objective summed over the batch, the nodes split, and one_sided's two counts."""
    fr = root_frontier(X, mean, resident=True)
    rows = []
    for depth in range(encoder.depths):
        total, C = depth_step(encoder, fr, args)
        child, side = centroid_children(fr, C)
        lone, held = one_sided(fr, side)
        rows.append(torch.stack([total, total.new_tensor(float(len(fr.bits))), lone.to(total), held.to(total)]))
        fr = child
    return torch.stack(rows, 1)


@torch.no_grad()
def evaluate(encoder, X, mean, args):
    """Per token of the encoder's depths [K]: the mse per pixel after it, and the share of nodes holding two or
    more of X's images that sent them all to one centroid; X routed down the encoder's tree."""
    encoder.eval()
    fr, mses, lones = root_frontier(X, mean, resident=True), [], []
    device = X.device
    for _ in range(encoder.depths):
        Nn = len(fr.bits)
        C = torch.cat([encoder.centroids(encoder.memory(*fr.history(torch.arange(i, min(i + args.node_batch, Nn)),
                                                                     device)))
                       for i in range(0, Nn, args.node_batch)])
        child, side = centroid_children(fr, C)
        lone, held = one_sided(fr, side)
        lones.append((lone / held.clamp_min(1)).item())
        fr = child
        mses.append(fr.R.square().mean().item())
    return mses, lones


def grow(encoder, opt, depths):
    """encoder.grow, and the optimizer's state for the position rows (zeros for the new rows, what zero
    gradients would have left them)."""
    p = encoder.position
    old = len(p)
    encoder.grow(depths)
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
    p.add_argument("--node-mean", action="store_true",
                   help="also show the network each node's predicted mean image (dvae/deltatree.py's HistoryReader)")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--batch", type=int, default=1024, help="training images per step")
    p.add_argument("--tf32", action="store_true", help="TF32 matmuls on GPUs that have them (~20%% faster on an L40S)")
    p.add_argument("--max-nodes", type=int, default=256,
                   help="nodes per forward pass; a depth spanning more is run in pieces with accumulated gradients")
    p.add_argument("--node-batch", type=int, default=512, help="nodes split at once when evaluating")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
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
        args.width, args.layers, args.heads, args.node_mean = cfg["width"], cfg["layers"], cfg["heads"], cfg["node_mean"]
    log(f"=== {time.strftime('%Y-%m-%d')} {vars(args)}")
    X = pixels(load_mnist("train", args.data_root)[:args.limit_train]).to(device)
    V = pixels(load_mnist("val", args.data_root)[:args.limit_val]).to(device)
    mean = X.mean(0) if ckpt is None else ckpt["mean"].to(device)
    K, N = args.depths if args.depths > 0 else math.inf, len(X)

    torch.manual_seed(args.seed)
    if ckpt is None:
        encoder = CentroidEncoder(1, args.width, args.layers, args.heads, node_mean=args.node_mean).to(device)
        if args.node_mean:
            with torch.no_grad():
                encoder.mean.copy_(mean)
    else:
        encoder = CentroidEncoder(**ckpt["shared"]["config"]).to(device)
        encoder.load_state_dict(ckpt["shared"]["state_dict"])
        assert encoder.depths <= K, f"--depths {K} is below the checkpoint's {encoder.depths}"
    log(f"centroid encoder: {sum(q.numel() for q in encoder.parameters()) / 1e6:.1f}M parameters, "
        f"{encoder.depths} depths, growing to {K}")

    opt = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history, step, first, best = [], 0, 0, (math.inf, -1)
    grown = 0.0  # when (in epochs) the encoder last grew
    if ckpt is not None:
        opt.load_state_dict(ckpt["optimizer"])
        for group in opt.param_groups:  # the settings given now
            group.update(lr=args.lr, initial_lr=args.lr, weight_decay=args.weight_decay)
        history, step, first = ckpt["metrics"], ckpt["step"], ckpt["epoch"] + 1
        for h in history:  # (from before train_mse was kept: the objective is the same thing)
            h.setdefault("train_mse", h["objective"][-1])
        grown = max(ckpt["grown"], first - args.grow_every)  # grow as it resumes, not to catch up
        if encoder.depths == K and ckpt.get("best"):
            best = tuple(ckpt["best"])  # the same full depth: early stopping carries on
        log(f"resumed {args.resume} after epoch {ckpt['epoch']} (step {step}), saving to {args.out}")
    step0 = step
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (step0 + s + 1) / args.warmup_steps))
    began = time.time() - (history[-1].get("seconds", 0.0) if history else 0.0)

    def save(epoch):
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save(dict(args=vars(args), mean=mean.cpu(), epoch=epoch, step=step, metrics=history, best=best,
                        shared=dict(config=encoder.config,
                                    state_dict={k: v.cpu() for k, v in encoder.state_dict().items()}),
                        optimizer=opt.state_dict(), grown=grown), args.out)

    def fmt(values, digits=5):
        return " ".join(f"{v:.{digits}f}" for v in values)

    for epoch in range(first, first + args.epochs):
        t0, last = time.time(), time.time()
        encoder.train()
        totals = torch.zeros(4, encoder.depths, device=device)  # (widened as the encoder grows)
        gen = torch.Generator().manual_seed(args.seed * 1_000_003 + epoch)
        batches, seen = list(torch.randperm(N, generator=gen).split(args.batch)), 0
        for i, idx in enumerate(batches):
            if encoder.depths < K and epoch + i / len(batches) - grown >= args.grow_every - 1e-9:
                grow(encoder, opt, encoder.depths + 1)
                grown += args.grow_every
            opt.zero_grad(set_to_none=True)
            stats = train_step(encoder, X[idx.to(device)], mean, args)
            if stats.shape[1] > totals.shape[1]:
                totals = torch.cat([totals, totals.new_zeros(4, stats.shape[1] - totals.shape[1])], 1)
            totals[:, :stats.shape[1]] += stats
            seen += len(idx)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if time.time() - last >= args.log_interval:
                last = time.time()
                log.detail(f"    step {step} ({seen}/{N} images this epoch)")
        t1 = time.time()
        val, val_lone = evaluate(encoder, V, mean, args)
        totals = totals[:, :encoder.depths]  # (a depth added mid-epoch counts only the images it saw)
        objective = (totals[0] / (N * D)).tolist()
        lone = (totals[2] / totals[3].clamp_min(1)).tolist()
        full = encoder.depths == K
        improved = full and val[-1] < best[0]
        if improved:
            best = (val[-1], epoch)
        history.append(dict(epoch=epoch, val_mse=val, train_mse=objective[-1], objective=objective, one_sided=lone,
                            val_one_sided=val_lone, trained=encoder.depths - 1, seconds=time.time() - began))
        plot_depths(history, args.plot or os.path.splitext(args.log)[0] + ".png")
        log(f"  epoch {epoch:2d}: val mse after {encoder.depths} tokens {val[-1]:.6f} (psnr {psnr(val[-1]):.2f} dB); "
            "per depth " + fmt(val) + "; objective per depth " + fmt(objective)
            + "; one-sided nodes per depth train " + fmt(lone, 3) + ", val " + fmt(val_lone, 3)
            + "; nodes per batch " + fmt((totals[1] / len(batches)).tolist(), 0)
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
