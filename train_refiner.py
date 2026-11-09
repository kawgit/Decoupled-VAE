"""Stage 1, refining variant (train_encoder.py trains the hyperplane encoder): train a refining delta tree with one
network for every depth (dvae/refine.py's Refiner). At each node the network runs Lloyd's algorithm on V centroids
itself, one iteration per call (each predicting the next centroids directly), and an image's token is the cell it
falls in; a code maps back to the network's predicted mean image at its node (the decoder, train_decoder.py, then
models the pixels around it). The tokens' prices are the prior's (train_prior.py); the shares the centroids imply
(dvae/refine.py) are only a diagnostic.

Every step sends a batch of training images down the tree as the encoder now splits it. At each depth, every
node the batch reaches runs the encoder's calls from its step-0 centroids (--init); each call's output is trained
on the actual means of the cells its input centroids make among the batch's images there (each cell weighted by
its image count, each node's error averaged over its calls, so every image counts once per depth), and the last
call's split, detached, takes the images to the next depth.

--init pca starts every node of a depth from plus and minus its top V/2 principal directions, scaled by their
rms: running averages of the batches' remainders' Gram matrix there, turned into eigenvectors every --refresh
steps (the next depth's tracked from before it joins). --init learned has the encoder's first call predict them
from the node's history alone. --steps calls follow. With --node-mean, the network also reads each node's
predicted mean image.

The encoder grows one depth every --grow-every epochs (1 by default; fractions grow mid-epoch), from 1 up to
--depths (Refiner.grow: new history positions and a new start), and every step trains all its depths. Every
depth's val mse is logged per epoch with the val bits per image under the encoder's implied shares, beside the
train mse after the last token (over the epoch's batches, each as training split it) and each call's val mse gain.
Early stopping watches the val mse after all --depths tokens once the encoder has grown to them, and stops a run
that goes a --patience-frac share of its epochs (at least --patience) without a better one (dvae/stopping.py).
A checkpoint holds everything training needs (the optimizer's state, the tracked starts, the step and the
history): saved every epoch while the encoder grows, then at each new best. --resume continues from one, growing
on to --depths if that is deeper. The progress plot (dvae/plots.py's plot_depths) is redrawn every epoch.

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
from dvae.data import SHAPES, get_device, load_images  # noqa: E402
from dvae.deltatree import by_nodes, pixels  # noqa: E402
from dvae.logs import RunLog  # noqa: E402
from dvae.plots import plot_depths  # noqa: E402
from dvae.refine import (CENTROID_INPUTS, QUERIES, Frontier, Refiner, Split, cat_splits, children,  # noqa: E402
                         choose, chunk_sse, root_frontier, route_path)
from dvae.stopping import add_patience_args, stagnant  # noqa: E402


def fold(grams, depth, R, ema):
    grams[depth].mul_(ema).add_(R.T @ R / len(R), alpha=1 - ema)


def pca_offsets(gram: torch.Tensor, V: int, previous: torch.Tensor | None = None) -> torch.Tensor:
    """Step-0 offsets [V, D] from a Gram matrix [D, D]: plus and minus each of its top V/2 eigenvectors scaled by
    the root of its eigenvalue, each keeping its sign from previous [V, D] (when given)."""
    gram = gram.double() if gram.device.type == "cuda" else gram.cpu().double()  # MPS has no float64
    values, vectors = torch.linalg.eigh(gram)
    out = []
    for j in range(V // 2):
        u = vectors[:, -1 - j].float().to(previous.device if previous is not None else "cpu")
        if previous is not None and u @ previous[2 * j] < 0:
            u = -u
        s = values[-1 - j].clamp_min(0).sqrt().float().item()
        out += [s * u, -s * u]
    return torch.stack(out)


@torch.no_grad()
def refresh(encoder: Refiner, grams, tracked):
    """Each tracked depth's step-0 offsets [n, V, D] from its running Gram matrix, and the encoder's from those."""
    for depth, gram in enumerate(grams):
        tracked[depth] = pca_offsets(gram, encoder.V, tracked[depth]).to(tracked)
    encoder.inits.copy_(tracked[:encoder.depths])


def depth_step(encoder: Refiner, fr: Frontier, args, full=True,
               weight: float | None = 1.0) -> tuple[torch.Tensor, Split]:
    """One depth of train_step, in pieces of at most --max-nodes nodes: the objective summed over fr's images
    (backward done) and every node's Split; full runs every call, else the depth's chosen number (chunk_sse).
    weight scales the loss and the objective (1 / the depth's chance of training this step, so their expected
    values are unchanged); None: the depth trains not at all this step, only routes the batch."""
    if weight is None:
        return fr.R.new_zeros(()), choose(encoder, fr, args.max_nodes, full)
    scale = weight / (args.batch * fr.R.shape[1])
    if fr.nodes <= args.max_nodes:  # one piece: no sorting or cutting, which would wait on the GPU
        loss, sse, _, s = chunk_sse(encoder, fr, None, args.grad_calls, full)
        (loss * scale).backward()
        return sse * weight, s
    order = torch.argsort(fr.node, stable=True)
    total = fr.R.new_zeros(())
    ids, parts = [], []
    for piece in by_nodes(order, fr.node[order], args.max_nodes):
        loss, sse, piece_nodes, s = chunk_sse(encoder, fr, piece, args.grad_calls, full)
        (loss * scale).backward()  # a fixed divisor, so every image weighs the same
        total = total + sse * weight
        ids.append(piece_nodes)
        parts.append(s)
    s = cat_splits(parts)
    whole = torch.argsort(torch.cat(ids))  # pieces hold whole nodes, in node order
    return total, Split(s.part[whole], s.cent[whole], s.w[whole])


def train_step(encoder: Refiner, X, mean, grams, args, full: list[bool] | None = None,
               weights: list[float | None] | None = None):
    """Send the batch X down every depth of the encoder as it splits them, accumulating each depth's gradient;
    with grams, fold the remainders each depth sees into its running Gram matrix (grams [n, D, D]), and the next
    depth's too while grams has one more. full[depth] (default all True) runs every call there, else only the
    depth's chosen number; weights[depth] (default 1) is depth_step's weight. Returns, per depth [K]: the
    objective, the token bits under the shares and the nodes split, summed over the batch; and the batch's
    squared error summed after its last token."""
    K = encoder.depths
    ahead = grams is not None and len(grams) > K
    fr = child = root_frontier(X, mean, resident=True)
    zero = X.new_zeros(())
    sses, bits, nodes = [zero] * K, [zero] * K, [zero] * K
    for depth in range(K):
        if grams is not None:
            fold(grams, depth, fr.R, args.ema)
        sses[depth], s = depth_step(encoder, fr, args, True if full is None else full[depth],
                                    1.0 if weights is None else weights[depth])
        nodes[depth] = X.new_full((), float(fr.nodes))  # (a fill, not a copy from the host, which would wait)
        child, _, token_bits = children(fr, s, encoder.V)
        bits[depth] = token_bits.sum()
        if depth + 1 < K or ahead:
            fr = child
    if ahead:
        assert grams is not None
        fold(grams, K, fr.R, args.ema)
    return torch.stack([torch.stack(sses), torch.stack(bits), torch.stack(nodes)]), child.R.detach().square().sum()


@torch.no_grad()
def evaluate(encoder: Refiner, X, mean, args):
    """Per token of the encoder's depths [K]: the mse per pixel after it, X routed down the encoder's tree; the
    mean bits per image it costs under the shares; and each call's gain, the mse per pixel it takes off the call
    before's split (the first's off no split)."""
    encoder.eval()
    fr, mses, bits, gains = root_frontier(X, mean, resident=True), [], [], []
    for _ in range(encoder.depths):
        s, errs = route_path(encoder, fr, args.node_batch)
        before = torch.cat([fr.R.square().sum(-1)[None], errs[:-1]])
        gains.append(((before - errs).mean(1) / fr.R.shape[1]).tolist())
        fr, _, token_bits = children(fr, s, encoder.V)
        mses.append(fr.R.square().mean().item())
        bits.append(token_bits.mean().item())
    return mses, bits, gains


def grow(encoder: Refiner, opt, depths, inits):
    """encoder.grow, and the optimizer's state for the position rows (zeros for the new rows, what zero
    gradients would have left them)."""
    p = encoder.position
    old = len(p)
    encoder.grow(depths, inits)
    for key, value in opt.state.get(p, {}).items():
        if torch.is_tensor(value) and value.shape[:1] == (old,):
            opt.state[p][key] = torch.cat([value, value.new_zeros(len(p) - old, *value.shape[1:])])


def psnr(mse):
    return -10 * math.log10(mse)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depths", type=int, default=10,
                   help="tokens per code, where the encoder stops growing (0: no cap; then nothing stops the run but "
                        "--epochs, as early stopping waits for the full depth)")
    p.add_argument("--grow-every", type=float, default=1.0,
                   help="epochs between depths added (fractions add them mid-epoch)")
    p.add_argument("--resume", default=None, metavar="CKPT",
                   help="a checkpoint to continue from (its architecture, V, steps and init are kept)")
    p.add_argument("-V", type=int, default=2, help="children per node")
    p.add_argument("--steps", "--lloyd-steps", type=int, default=16,
                   help="Lloyd iterations (network calls) per node after step 0")
    p.add_argument("--init", choices=("pca", "learned"), default="pca", help="step 0's centroids")
    p.add_argument("--centroid-in", choices=CENTROID_INPUTS, default="unit",
                   help="how a call reads the current centroids: each as a unit direction and log length (unit), "
                        "as raw offsets and log length (raw), or unit ones plus a token per pair for the "
                        "hyperplane between their cells (pairs)")
    p.add_argument("--query", choices=QUERIES, default="tokens",
                   help="a token per centroid (tokens), or one token summing the hyperplanes between them, read by "
                        "one head for all V centroids, with no step embedding (single, dvae/deltatree.py's query "
                        "for V = 2)")
    p.add_argument("--siblings", action="store_true",
                   help="each history token also reads every child's offset at that depth (what dvae/deltatree.py's "
                        "side and m tell about the other side, for any V)")
    p.add_argument("--tower", action="store_true",
                   help="with --query single and V = 2, output dvae/deltatree.py's r and m (children r and -m r) "
                        "instead of free centroids (an ablation)")
    p.add_argument("--node-mean", action="store_true",
                   help="the network also reads each node's predicted mean image")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--batch", type=int, default=1024, help="training images per step")
    p.add_argument("--tf32", action="store_true", help="TF32 matmuls on GPUs that have them (~20%% faster on an L40S)")
    p.add_argument("--dynamic-steps", action="store_true",
                   help="each depth uses the number of steps n (at most --steps) with the best discounted val gain, "
                        "cumulative gain(n) * (1 - --step-discount)^n, re-chosen every epoch; training runs all steps "
                        "at a depth with a probability that halves every --full-half-life epochs from 1 when it joins")
    p.add_argument("--step-discount", type=float, default=0.001)
    p.add_argument("--gain-smoothing", type=float, default=0.8,
                   help="the weight a running average of each depth's per-step val gains keeps on the past each epoch "
                        "(the steps are chosen from it; a new depth starts from its first measurement)")
    p.add_argument("--full-half-life", type=float, default=4.0,
                   help="epochs over which a depth's chance of training every step halves (--dynamic-steps)")
    p.add_argument("--grad-prob-min", type=float, default=1.0,
                   help="train each depth on a step with a chance rising linearly from this at depth 1 to 1 at the "
                        "deepest (its loss weighted by 1 / the chance, so the expected gradient is unchanged; the "
                        "other steps only route through it); 1: every depth every step")
    p.add_argument("--grad-calls", type=int, default=0,
                   help="backpropagate through only this many of a node's calls, drawn at random per piece (all "
                        "calls still run, so partitions are unchanged; 0: all)")
    p.add_argument("--bf16", action="store_true",
                   help="run the network under bfloat16 autocast (centroids, partitions and losses stay float32)")
    p.add_argument("--channels-last", action="store_true", help="the centroid head's convolutions in channels_last")
    p.add_argument("--graphs", action="store_true",
                   help="route without autograd, each call a CUDA graph's replay, then train the chosen calls in one "
                        "batch (the same loss and gradient, far fewer kernel launches)")
    p.add_argument("--recompute", action="store_true",
                   help="recompute each call in the backward pass instead of keeping its activations (less memory)")
    p.add_argument("--max-nodes", type=int, default=256,
                   help="nodes per piece of a depth's training step (memory grows with the nodes queried)")
    p.add_argument("--node-batch", type=int, default=512, help="nodes split at once when evaluating")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--ema", type=float, default=0.95, help="running Gram matrices' decay per step (--init pca)")
    p.add_argument("--refresh", type=int, default=50, help="steps between start updates (--init pca)")
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
    p.add_argument("--dataset", choices=sorted(SHAPES), default="mnist")
    args = p.parse_args()
    args.out, args.log = args.out or artifact("model.pt"), args.log or artifact("run.log")

    device = get_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    log = RunLog(args.log, device)

    ckpt = torch.load(args.resume, map_location=device, weights_only=False) if args.resume else None
    if ckpt is not None:  # the architecture is the checkpoint's
        cfg = ckpt["encoder"]["config"]
        args.width, args.layers, args.heads, args.node_mean = cfg["width"], cfg["layers"], cfg["heads"], cfg["node_mean"]
        args.V, args.steps, args.init = cfg["V"], cfg["steps"], cfg["init"]
        args.centroid_in, args.query = cfg.get("centroid_in", "unit"), cfg.get("query", "tokens")
        args.tower, args.siblings = cfg.get("tower", False), cfg.get("siblings", False)
        shape = tuple(cfg.get("shape", (1, 28, 28)))
        args.dataset = next(name for name, known in SHAPES.items() if known == shape)
    log(f"=== {time.strftime('%Y-%m-%d')} train_refiner.py {dict(vars(args), device=device)}")
    X = pixels(load_images(args.dataset, "train", args.data_root)[:args.limit_train]).to(device)
    Xv = pixels(load_images(args.dataset, "val", args.data_root)[:args.limit_val]).to(device)
    D = X.shape[1]
    mean = X.mean(0) if ckpt is None else ckpt["mean"].to(device)
    K, N = args.depths if args.depths > 0 else math.inf, len(X)
    R = X - mean
    root_gram = R.T @ R / N
    pca = args.init == "pca"
    root_offsets = pca_offsets(root_gram, args.V).to(device) if pca else torch.zeros(args.V, D, device=device)
    del R

    torch.manual_seed(args.seed)
    if ckpt is None:
        encoder = Refiner(1, args.V, args.steps, args.init, args.width, args.layers, args.heads,
                          args.node_mean, args.centroid_in, args.query, args.tower, args.siblings,
                          SHAPES[args.dataset]).to(device)
        with torch.no_grad():
            encoder.inits[0] = root_offsets
            if args.node_mean:
                encoder.mean.copy_(mean)
    else:
        encoder = Refiner(**ckpt["encoder"]["config"]).to(device)
        encoder.load_state_dict(ckpt["encoder"]["state_dict"])
        assert encoder.depths <= K, f"--depths {K} is below the checkpoint's {encoder.depths}"
    encoder.speed(args.recompute, args.bf16, args.channels_last, args.graphs)
    # each depth's tracked step-0 offsets, the encoder's and the next one's (the root's until tracked)
    tracked = torch.cat([encoder.inits.detach().clone(), root_offsets[None]])[:min(encoder.depths + 1, K)]
    grams = None
    if pca:
        grams = root_gram.expand(len(tracked), D, D).clone()
        if ckpt is not None and ckpt.get("grams") is not None:
            kept = ckpt["grams"].to(device)[:len(tracked)]
            grams[:len(kept)] = kept
            tracked[:len(kept)] = ckpt["tracked"].to(device)[:len(kept)]
    log(f"encoder: {sum(q.numel() for q in encoder.parameters()) / 1e6:.1f}M parameters, V = {encoder.V}, "
        f"{encoder.steps} steps from {encoder.init} starts, {encoder.depths} depths, growing to {K}")

    opt = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history, step, first, best = [], 0, 0, (math.inf, -1)
    grown = 0.0  # when (in epochs) the encoder last grew
    joined = [0.0] * encoder.depths  # when (in epochs) each depth joined
    smoothed: list[list[float]] = []  # each depth's running average of its per-step val gains
    if ckpt is not None:
        opt.load_state_dict(ckpt["optimizer"])
        for group in opt.param_groups:  # the settings given now
            group.update(lr=args.lr, initial_lr=args.lr, weight_decay=args.weight_decay)
        history, step, first = ckpt["metrics"], ckpt.get("step", 0), ckpt["epoch"] + 1
        grown = max(ckpt.get("grown", -math.inf), first - args.grow_every)  # grow as it resumes, not to catch up
        joined = (list(ckpt.get("joined") or []) + [0.0] * encoder.depths)[:encoder.depths]
        smoothed = list(ckpt.get("smoothed") or [])
        if encoder.depths == K and ckpt.get("best"):
            best = tuple(ckpt["best"])  # the same full depth: early stopping carries on
        log(f"resumed {args.resume} after epoch {ckpt['epoch']} (step {step}), saving to {args.out}")
    step0 = step
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (step0 + s + 1) / args.warmup_steps))
    began = time.time() - (history[-1].get("seconds", 0.0) if history else 0.0)

    def save(epoch):
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save(dict(args=vars(args), mean=mean.cpu(), epoch=epoch, step=step, metrics=history, best=best,
                        encoder=dict(config=encoder.config,
                                     state_dict={k: v.cpu() for k, v in encoder.state_dict().items()}),
                        optimizer=opt.state_dict(), tracked=tracked.cpu(), grown=grown, joined=joined,
                        smoothed=smoothed,
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
        if depth + 1 < K:
            tracked = torch.cat([tracked, root_offsets[None]])
            if grams is not None:
                grams = torch.cat([grams, root_gram[None]])

    for epoch in range(first, first + args.epochs):
        t0, last = time.time(), time.time()
        encoder.train()
        totals = torch.zeros(3, encoder.depths, device=device)  # (widened as the encoder grows)
        last_sse = torch.zeros((), device=device)  # after each batch's last token
        gen = torch.Generator().manual_seed(args.seed * 1_000_003 + epoch)
        batches, seen = list(torch.randperm(N, generator=gen).split(args.batch)), 0
        for i, idx in enumerate(batches):
            now = epoch + i / len(batches)
            if encoder.depths < K and now - grown >= args.grow_every - 1e-9:
                add_depth()
                grown += args.grow_every
                joined.append(now)
            every = None
            if args.dynamic_steps:  # each depth runs every step with its chance, else its chosen number
                chance = [0.5 ** ((now - j) / args.full_half_life) for j in joined]
                every = (torch.rand(len(chance), generator=gen) < torch.tensor(chance)).tolist()
            weights = None
            if args.grad_prob_min < 1:  # shallower depths train on fewer steps, their loss weighted up to match
                k = encoder.depths
                odds = [args.grad_prob_min + (1 - args.grad_prob_min) * d / max(k - 1, 1) for d in range(k)]
                draws = torch.rand(k, generator=gen).tolist()
                weights = [1 / q if u < q else None for q, u in zip(odds, draws)]
            opt.zero_grad(set_to_none=True)
            stats, sse = train_step(encoder, X[idx.to(device)], mean, grams, args, every, weights)
            last_sse += sse
            if stats.shape[1] > totals.shape[1]:
                totals = torch.cat([totals, totals.new_zeros(3, stats.shape[1] - totals.shape[1])], 1)
            totals[:, :stats.shape[1]] += stats
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
        val, val_bits, call_gains = evaluate(encoder, Xv, mean, args)
        totals = totals[:, :encoder.depths]  # (a depth added mid-epoch counts only the images it saw)
        objective = (totals[0] / (N * D)).tolist()
        train_bits = (totals[1] / N).tolist()
        train_mse = last_sse.item() / (N * D)
        full = encoder.depths == K
        improved = full and val[-1] < best[0]
        if improved:
            best = (val[-1], epoch)
        stops = [encoder.stop_at(d) for d in range(encoder.depths)]  # what val was routed with
        b = args.gain_smoothing  # each depth's gains averaged over epochs; a new depth starts from its first
        smoothed = [[b * a + (1 - b) * x for a, x in zip(smoothed[d], g)] if d < len(smoothed) else list(g)
                    for d, g in enumerate(call_gains)]
        keep = 1 - args.step_discount  # the steps the averaged gains choose (next epoch's, with --dynamic-steps)
        chosen = [max(range(1, len(g) + 1), key=lambda n: sum(g[:n]) * keep ** n) for g in smoothed]
        history.append(dict(epoch=epoch, val_mse=val, train_mse=train_mse, objective=objective, val_bits=val_bits,
                            train_bits=train_bits, step_gains=call_gains, smoothed_gains=smoothed, stops=stops,
                            chosen=chosen,
                            trained=encoder.depths - 1, seconds=time.time() - began))
        plot_depths(history, args.plot or os.path.splitext(args.log)[0] + ".png")
        log(f"  epoch {epoch:2d}: val mse after {encoder.depths} tokens {val[-1]:.6f} (psnr {psnr(val[-1]):.2f} dB); "
            "per depth " + fmt(val) + f"; train mse {train_mse:.6f}; val bits per image under the shares "
            f"{sum(val_bits):.3f} (per depth " + fmt(val_bits, 3) + f"), train {sum(train_bits):.3f}; objective per "
            "depth " + fmt(objective) + "; nodes per batch " + fmt((totals[2] / len(batches)).tolist(), 0)
            + f" ({t1 - t0:.0f}s train, {time.time() - t1:.0f}s val){'  *' if improved else ''}")
        if args.dynamic_steps:
            chance = [0.5 ** ((epoch + 1 - j) / args.full_half_life) for j in joined]
            log(f"    steps used per depth {' '.join(map(str, stops))}; chance of every step "
                + fmt(chance, 2))
        if not full or improved:
            save(epoch)  # the latest while growing, then the best (with the steps val was routed with)
        elif stagnant(epoch, best[1], args):
            break
        if args.dynamic_steps:  # next epoch's: the best discounted cumulative val gain at each depth
            encoder.set_stops(chosen)
    if best[1] < 0:
        log(f"stopped at {encoder.depths} of {K} depths; the latest state is in {args.out}")
        return
    log(f"kept epoch {best[1]}: val mse after {K} tokens {best[0]:.6f} (psnr {psnr(best[0]):.2f} dB), saved {args.out}")


if __name__ == "__main__":
    main()
