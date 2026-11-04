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

With --gate, depths join training one epoch at a time: epoch e trains depths 0..min(e, --depths - 1) only, and
the batch's split at the deepest of them gives the next depth's remainders, which track its start before it
trains.

With --chunk C, batches after the first epoch follow the tree: the training images, sorted by the codes the
last epoch gave them (token 0 first, ties broken at random), are cut into runs of C, the runs shuffled, and
consecutive runs joined into batches, so a batch's images share more tokens and reach fewer nodes (a node's
cost does not depend on its images). C = --batch makes each batch one run; smaller C mixes more of the tree.

Early stopping watches the held-out (val) images' mse after all --depths tokens, routed the same way, once every
depth trains; every depth's val mse is logged.
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
from dvae.deltatree import (D, SharedDecoder, Split, by_nodes, cat_splits, children, chunk_sse,  # noqa: E402
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


def depth_step(decoder, fr, args):
    """One depth of train_step, in pieces of at most --max-nodes nodes: the objective and PCA error summed over
    fr's images (backward done) and every node's Split."""
    order, _ = node_order(fr)
    total, pca_total = fr.R.new_zeros(()), fr.R.new_zeros(())
    ids, parts = [], []
    for piece in by_nodes(order, fr.node[order], args.max_nodes):
        sse, pca, _, piece_nodes, s = chunk_sse(decoder, fr, piece, args.lloyd_steps, tol=args.tol,
                                                dir_weight=args.dir_weight)
        (sse / (args.batch * D)).backward()  # a fixed divisor, so every image weighs the same
        total, pca_total = total + sse.detach(), pca_total + pca
        ids.append(piece_nodes)
        parts.append(s)
    s = cat_splits(parts)
    whole = torch.argsort(torch.cat(ids))  # pieces hold whole nodes, in node order
    return total, pca_total, Split(s.n[whole], s.b[whole], s.r[whole], s.m[whole],
                                   None if s.u is None else s.u[whole], s.steps[whole])


def train_step(decoder, X, mean, grams, args, active, graphs: GraphedDepths | None = None):
    """Send the batch X down depths 0..active as decoder splits them, accumulating each depth's gradient; without
    seeds, fold the remainders each depth sees into its running Gram matrix (grams [K, D, D]), and depth
    active + 1's too. Returns, per depth [K] (0 past active): the objective and the PCA reconstruction error
    summed over the batch, and the Lloyd steps taken and nodes split; and each image's key (child_key) at the
    deepest depth it reached [b], on the CPU. With graphs, a full batch's depths run as CUDA graphs."""
    K = args.depths
    fr = root_frontier(X, mean, resident=True)
    zero = X.new_zeros(())
    sses, pcas, steps, nodes = [zero] * K, [zero] * K, [zero] * K, [zero] * K
    for depth in range(active + 1):
        R = fr.R
        if grams is not None:
            fold(grams, depth, R, args.ema)
        if graphs is not None and len(X) == graphs.batch:
            total, s = graphs.run(fr)
            pca_total = zero
        else:
            total, pca_total, s = depth_step(decoder, fr, args)
        sses[depth], pcas[depth] = total, pca_total
        steps[depth], nodes[depth] = s.steps.sum(), R.new_tensor(float(len(s.steps)))
        if depth + 1 < K:
            fr, _ = children(fr, s)
            if depth == active and grams is not None:
                fold(grams, depth + 1, fr.R, args.ema)  # the next depth's start, watched before it trains
    return torch.stack([torch.stack(sses), torch.stack(pcas), torch.stack(steps), torch.stack(nodes)]), \
        fr.bits[fr.node.to(fr.bits.device)].cpu()


@torch.no_grad()
def refresh(decoder, grams):
    """Each depth's start from its running Gram matrix; a start keeps its sign from before."""
    for depth, gram in enumerate(grams):
        gram = gram.double() if gram.device.type == "cuda" else gram.cpu().double()  # MPS has no float64
        top = torch.linalg.eigh(gram)[1][:, -1].float().to(decoder.starts.device)
        decoder.starts[depth] = top if top @ decoder.starts[depth] >= 0 else -top


@torch.no_grad()
def evaluate(decoder, X, mean, args):
    """Per token [K]: the mse per pixel after it, X routed down the decoder's tree; the mean Lloyd steps per
    node; and the share of nodes that took all --lloyd-steps."""
    decoder.eval()
    fr, mses, steps, capped = root_frontier(X, mean, resident=True), [], [], []
    for _ in range(args.depths):
        fr, _, s = split(decoder, fr, args.lloyd_steps, args.node_batch, args.tol)
        mses.append(fr.R.square().mean().item())
        steps.append(s.steps.mean().item())
        capped.append((s.steps >= args.lloyd_steps).float().mean().item())
    return mses, steps, capped


def psnr(mse):
    return -10 * math.log10(mse)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depths", type=int, default=10, help="tokens per code")
    p.add_argument("--lloyd-steps", type=int, default=4, help="Lloyd steps (with --tol, the most a node takes)")
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
    p.add_argument("--gate", action="store_true", help="epoch e trains depths 0..e only")
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

    log(f"=== {time.strftime('%Y-%m-%d')} {vars(args)}")
    X = pixels(load_mnist("train", args.data_root)[:args.limit_train]).to(device)
    V = pixels(load_mnist("val", args.data_root)[:args.limit_val]).to(device)
    mean = X.mean(0)
    K, N = args.depths, len(X)

    torch.manual_seed(args.seed)
    decoder = SharedDecoder(K, args.width, args.layers, args.heads, seeded=args.seeded).to(device)
    R = X - mean
    grams = None
    with torch.no_grad():
        decoder.starts.copy_(top_direction(R).expand(K, D))  # the root's start; the depths' first, without seeds
    if not args.seeded and args.start == "depth":
        grams = (R.T @ R / N).expand(K, D, D).clone()
    log(f"shared decoder: {sum(q.numel() for q in decoder.parameters()) / 1e6:.1f}M parameters, {K} depths")

    graphs = None
    if args.graphs:
        assert device.type == "cuda" and not args.seeded and args.tol == 0, "--graphs: CUDA, no --seeded or --tol"
        graphs = GraphedDepths(decoder, args.batch, args.lloyd_steps, args.batch * D)
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup_steps))
    gen = torch.Generator().manual_seed(args.seed)
    best: tuple[float, int, dict | None] = (math.inf, -1, None)
    history, step, began = [], 0, time.time()
    active = K - 1  # the deepest depth trained
    codes = None  # each training image's code in the last epoch

    def save(state, epoch):
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save(dict(args=vars(args), mean=mean.cpu(), epoch=epoch, metrics=history,
                        shared=dict(config=decoder.config, state_dict={k: v.cpu() for k, v in state.items()})),
                   args.out)

    def fmt(values, digits=5):
        return " ".join(f"{v:.{digits}f}" for v in values)

    for epoch in range(args.epochs):
        active = min(epoch, K - 1) if args.gate else K - 1
        t0, last = time.time(), time.time()
        decoder.train()
        totals = torch.zeros(4, K, device=device)
        batches, seen = epoch_batches(N, codes, args, gen), 0
        codes = torch.zeros(N, dtype=torch.int64)
        for idx in batches:
            opt.zero_grad(set_to_none=graphs is None)  # graphs add into the gradients' fixed buffers
            stats, keys = train_step(decoder, X[idx.to(device)], mean, grams, args, active, graphs)
            totals += stats
            codes[idx] = keys
            seen += len(idx)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if grams is not None and step % args.refresh == 0:
                refresh(decoder, grams)
            if time.time() - last >= args.log_interval:
                last = time.time()
                log(f"    step {step} ({seen}/{N} images this epoch)")
        t1 = time.time()
        val, val_steps, capped = evaluate(decoder, V, mean, args)
        objective = (totals[0] / (N * D)).tolist()
        pca = (totals[1] / (N * D)).tolist()
        train_steps = (totals[2] / totals[3].clamp_min(1)).tolist()
        trained = active
        history.append(dict(epoch=epoch, val_mse=val, objective=objective, pca=pca, train_steps=train_steps,
                            val_steps=val_steps, val_capped=capped, trained=trained,
                            seconds=time.time() - began))
        plot_depths(history, args.plot or os.path.splitext(args.log)[0] + ".png")
        improved = trained == K - 1 and val[-1] < best[0]
        log(f"  epoch {epoch:2d}: val mse after {K} tokens {val[-1]:.6f} (psnr {psnr(val[-1]):.2f} dB); per depth "
            + fmt(val) + "; objective per depth " + fmt(objective)
            + (("; pca per depth " + fmt(pca)) if args.seeded else "")
            + "; lloyd steps per depth train " + fmt(train_steps, 2) + ", val " + fmt(val_steps, 2)
            + ", val capped " + fmt(capped, 2) + "; nodes per batch " + fmt((totals[3] / len(batches)).tolist(), 0)
            + (f"; trained depths 0..{trained}" if args.gate else "")
            + f" ({t1 - t0:.0f}s train, {time.time() - t1:.0f}s val){'  *' if improved else ''}")
        if trained < K - 1:
            save(decoder.state_dict(), epoch)  # the latest, until every depth trains
            continue
        if improved:
            best = (val[-1], epoch, {k: v.detach().clone() for k, v in decoder.state_dict().items()})
            save(best[2], epoch)
        elif epoch - best[1] >= args.patience:
            break
    if best[2] is None:
        log(f"stopped before every depth trained (depths 0..{active}); the latest weights are in {args.out}")
        return
    log(f"kept epoch {best[1]}: val mse after {K} tokens {best[0]:.6f} (psnr {psnr(best[0]):.2f} dB), saved {args.out}")


if __name__ == "__main__":
    main()
