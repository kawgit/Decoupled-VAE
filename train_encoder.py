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
without a better one (dvae/stopping.py); every depth's val mse is logged, beside the train mse after the last token
(over the epoch's batches, each as training split it). A checkpoint holds everything training needs (the
optimizer's state, the tracked starts, the step and the history): saved every epoch while the encoder grows, then
at each new best. --resume continues from one, growing on to --depths if that is deeper. The progress plot
(dvae/plots.py's plot_depths) is redrawn every epoch.

Next: dump_encoder.py writes every image's code for the prior and the decoder.
"""

import argparse
import itertools
import math
import os
import time

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")  # ~6.5 GB: fail instead of swapping
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")

import torch  # noqa: E402

from dvae.artifacts import artifact  # noqa: E402
from dvae.data import SHAPES, get_device, load_images  # noqa: E402
from dvae.logs import RunLog  # noqa: E402
from dvae.deltatree import (Encoder, Split, by_nodes, cat_splits, children, choose, chunk_sse,  # noqa: E402
                            lloyd, node_order, pixels, root_frontier, split, top_direction)
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


def depth_step(encoder, fr, args, steps, weight: float | None = 1.0, clock: "DepthClock | None" = None):
    """One depth of train_step, steps Lloyd steps at its nodes, in pieces of at most --max-nodes nodes: the
    objective and PCA error summed over fr's images (backward done) and every node's Split. weight scales the
    loss and both sums (1 / the depth's chance of training this step, so their expected values are unchanged);
    None: the depth trains not at all this step, only splits the batch."""
    if weight is None:
        zero = fr.R.new_zeros(())
        return zero, zero, choose(encoder, fr, steps, args.max_nodes, args.tol)
    grad_steps, tilt = getattr(args, "grad_steps", 0), getattr(args, "grad_tilt", None)
    tilt = None if tilt is None or tilt < 0 else tilt

    def backward(loss):
        began = None if clock is None else clock.start()
        loss.backward()
        if clock is not None:
            clock.stop(fr.depth, began, "backward")

    if len(fr.bits) <= args.max_nodes:  # one piece: no sorting or cutting, which would wait on the GPU
        sse, pca, _, _, s = chunk_sse(encoder, fr, None, steps, tol=args.tol, dir_weight=args.dir_weight,
                                      grad_steps=grad_steps, grad_tilt=tilt, clock=clock)
        backward(sse * (weight / (args.batch * fr.R.shape[1])))
        return sse.detach() * weight, pca * weight, s
    order, _ = node_order(fr)
    total, pca_total = fr.R.new_zeros(()), fr.R.new_zeros(())
    ids, parts = [], []
    for piece in by_nodes(order, fr.node[order], args.max_nodes):
        sse, pca, _, piece_nodes, s = chunk_sse(encoder, fr, piece, steps, tol=args.tol, dir_weight=args.dir_weight,
                                                grad_steps=grad_steps, grad_tilt=tilt, clock=clock)
        backward(sse * (weight / (args.batch * fr.R.shape[1])))  # a fixed divisor: every image weighs the same
        total, pca_total = total + sse.detach() * weight, pca_total + pca * weight
        ids.append(piece_nodes)
        parts.append(s)
    s = cat_splits(parts)
    whole = torch.argsort(torch.cat(ids))  # pieces hold whole nodes, in node order
    return total, pca_total, Split(s.n[whole], s.b[whole], s.r[whole], s.m[whole],
                                   None if s.u is None else s.u[whole], s.steps[whole])


def nonneg_fit(feats: list[list[float]], ys: list[float]) -> list[float]:
    """Least-squares coefficients of ys on feats (rows of features) kept nonnegative: a feature whose
    coefficient would be negative is dropped and the rest refit."""
    X, y = torch.tensor(feats, dtype=torch.float64), torch.tensor(ys, dtype=torch.float64)
    keep = list(range(X.shape[1]))
    coef = torch.zeros(len(keep), dtype=torch.float64)
    while keep:
        coef = torch.linalg.lstsq(X[:, keep], y[:, None]).solution[:, 0]
        if (coef >= 0).all() or len(keep) == 1:
            break
        keep.pop(int(coef.argmin()))
    out = [0.0] * X.shape[1]
    for j, c in zip(keep, coef.clamp_min(0).tolist()):
        out[j] = c
    return out


class DepthClock:
    """GPU seconds each depth's training takes, by part (labels: the whole depth, its memory, its Lloyd steps'
    forward pass, its backward pass, its routing), timed with CUDA events (nothing waits until totals) or, on the
    CPU, the host clock."""

    def __init__(self, cuda: bool):
        self.cuda, self.marks = cuda, []

    def start(self):
        if not self.cuda:
            return time.perf_counter()
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def stop(self, depth, began, label="depth"):
        self.marks.append((label, depth, began, self.start()))

    def totals(self, K) -> dict[str, list[float]]:
        """Each label's seconds per depth since the last totals {label: [K]}."""
        if self.cuda:
            torch.cuda.synchronize()
        out = {label: [0.0] * K for label in ("depth", "memory", "steps", "backward", "route")}
        for label, depth, a, b in self.marks:
            out[label][depth] += a.elapsed_time(b) / 1000 if self.cuda else b - a
        self.marks = []
        return out


def train_step(encoder, X, mean, grams, args, caps, weights: list[float | None] | None = None,
               clock: DepthClock | None = None):
    """Send the batch X down every depth of the encoder as it splits them (depth d with caps[d] Lloyd steps),
    accumulating each depth's gradient;
    without seeds, fold the remainders each depth sees into its running Gram matrix (grams [n, D, D]), and the
    next depth's too while grams has one more (it tracks its start before it joins). Returns, per depth [K]: the
    objective and the PCA reconstruction error summed over the batch, and the Lloyd steps taken and nodes split;
    the batch's squared error summed after its last token (the train mse's numerator, as val's is measured);
    and each image's key (child_key) at the deepest depth it reached [b], on the CPU."""
    K = encoder.depths
    ahead = grams is not None and len(grams) > K
    fr = root_frontier(X, mean, resident=True)
    child = fr
    zero = X.new_zeros(())
    sses, pcas, steps, nodes = [zero] * K, [zero] * K, [zero] * K, [zero] * K
    for depth in range(K):
        R = fr.R
        if grams is not None:
            fold(grams, depth, R, args.ema)
        began = None if clock is None else clock.start()
        total, pca_total, s = depth_step(encoder, fr, args, caps[depth], 1.0 if weights is None else weights[depth],
                                         clock)
        if clock is not None:
            clock.stop(depth, began)
        sses[depth], pcas[depth] = total, pca_total
        steps[depth] = s.steps.sum()
        nodes[depth] = R.new_full((), float(len(s.steps)))  # (a fill, not a copy from the host, which would wait)
        routed = None if clock is None else clock.start()
        child, _ = children(fr, s)
        if clock is not None:
            clock.stop(depth, routed, "route")
        if depth + 1 < K or ahead:
            fr = child
    if ahead:
        assert grams is not None
        fold(grams, K, fr.R, args.ema)
    return torch.stack([torch.stack(sses), torch.stack(pcas), torch.stack(steps), torch.stack(nodes)]), \
        child.R.detach().square().sum(), fr.bits[fr.node.to(fr.bits.device)].cpu()


@torch.no_grad()
def refresh(encoder, grams, tracked, power=0):
    """Each tracked depth's start [n, D] from its running Gram matrix (a start keeps its sign from before), and
    the encoder's depths' starts from those. With power > 0, the top eigenvector comes from that many power
    iterations from the start as it was (cheap for large images, where a full eigendecomposition per depth each
    refresh is not), else from an eigendecomposition."""
    if power > 0:
        top = tracked[:len(grams)].clone()
        for _ in range(power):
            top = torch.einsum("kde,ke->kd", grams, top)
            top = top / top.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        tracked[:len(grams)] = top  # (from the old start, its sign is kept)
    else:
        for depth, gram in enumerate(grams):
            gram = gram.double() if gram.device.type == "cuda" else gram.cpu().double()  # MPS has no float64
            top = torch.linalg.eigh(gram)[1][:, -1].float().to(tracked.device)
            tracked[depth] = top if top @ tracked[depth] >= 0 else -top
    encoder.starts.copy_(tracked[:encoder.depths])


@torch.no_grad()
def lloyd_path(encoder, fr, steps, batch, use=None):
    """Every node's Split after use (default steps) Lloyd steps, and every image's squared error after the split
    each of all steps gives [steps, N]."""
    use = steps if use is None else use
    device, Nn = fr.R.device, len(fr.bits)
    P = fr.R.shape[1]
    n, b, r, m = fr.R.new_zeros(Nn, P), fr.R.new_zeros(Nn), fr.R.new_zeros(Nn, P), fr.R.new_zeros(Nn)
    errs = fr.R.new_zeros(steps, len(fr.R))
    for i in range(0, Nn, batch):
        nodes = torch.arange(i, min(i + batch, Nn), device=device)
        mine = (fr.node >= i) & (fr.node < i + len(nodes))
        R, node = fr.R[mine], fr.node[mine] - i
        mems = encoder.memory(*fr.history(nodes, device))
        for t, (_, tn, tb, tr, tm, _) in enumerate(lloyd(encoder, mems, fr.starts(encoder, nodes), steps)):
            rare = (R * tn[node]).sum(-1) >= tb[node]
            errs[t, mine] = (R - torch.where(rare[:, None], tr[node], -tm[node, None] * tr[node])).square().sum(-1)
            if t == use - 1:
                n[nodes], b[nodes], r[nodes], m[nodes] = tn, tb, tr, tm
    return Split(n, b, r, m, None, fr.R.new_full((Nn,), float(use))), errs


@torch.no_grad()
def evaluate(encoder, X, mean, args, caps, probe: list[int] | None = None):
    """Per token of the encoder's depths [K]: the mse per pixel after it, X routed down the encoder's tree
    (depth d with caps[d] Lloyd steps); the mean Lloyd steps per node; the share of nodes that took all their
    steps; and each step's gain, the mse per pixel it takes off the step before's split (the first's off no
    split; none with --tol), for every step up to probe[d] (default caps[d]) though routing uses caps[d]."""
    encoder.eval()
    fr, mses, steps, capped, gains = root_frontier(X, mean, resident=True), [], [], [], []
    for depth in range(encoder.depths):
        if args.tol > 0:
            fr, _, s = split(encoder, fr, caps[depth], args.node_batch, args.tol)
            gains.append([])
        else:
            s, errs = lloyd_path(encoder, fr, probe[depth] if probe else caps[depth], args.node_batch, caps[depth])
            before = torch.cat([fr.R.square().sum(-1)[None], errs[:-1]])
            gains.append(((before - errs).mean(1) / fr.R.shape[1]).tolist())
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
    p.add_argument("--dynamic-steps", action="store_true",
                   help="each depth uses the Lloyd steps n (at most --lloyd-steps) with the best gain(n) - cost * n: "
                        "gain(n) its cumulative val mse gain after n steps (averaged over epochs), cost one step's "
                        "price, val mse's fall per epoch over the last --rate-window epochs (at the deepest depth the "
                        "run had then) times --cost-window epochs, times the share of training time one step at that "
                        "depth took; re-chosen every "
                        "epoch, with no cap (--lloyd-steps is a new depth's first number). To see past it, a depth "
                        "trains and is measured with explore(n) = max(n + 1, ceil(1.5 n)) steps, training so with a "
                        "chance that halves every --full-half-life epochs from 1 when it joins")
    p.add_argument("--step-threshold", type=float, default=0.0,
                   help="the threshold schedule (instead of --dynamic-steps/--dynamic-growth), in val mse per GPU "
                        "second: each depth uses the n Lloyd steps (at least 1) with the most cumulative val mse gain "
                        "(averaged over epochs) less n times this threshold times one step's cost there (its GPU "
                        "seconds per epoch, timed); a depth is added while the deepest one's first step's gain beats "
                        "the threshold times its cost; and "
                        "the threshold halves once training slows under it: when the train mse's fall per GPU second "
                        "over the last --rate-epochs epochs drops below --slowdown times its best since the "
                        "threshold was set. Depths and steps measure and explore past their counts as with "
                        "--dynamic-steps. 0: off")
    p.add_argument("--rate-epochs", type=int, default=3,
                   help="epochs over which the threshold schedule measures how fast train mse falls")
    p.add_argument("--slowdown", type=float, default=0.5,
                   help="the share of its best rate under a threshold below which the rate halves the threshold")
    p.add_argument("--cost-window", type=int, default=20,
                   help="the horizon, in epochs, over which a step's or a depth's training time is priced")
    p.add_argument("--rate-window", type=int, default=5,
                   help="recent epochs whose val mse fall per epoch gives the price of training time")
    p.add_argument("--dynamic-growth", action="store_true",
                   help="grow a depth (at an epoch's end, in place of --grow-every) once the next token's expected val "
                        "gain beats its price: the last token's gain times the ratio of the last two (at most 1), "
                        "against a step's price (see --dynamic-steps) with the share of training time the deepest "
                        "depth now takes; with --depths 0, without end")
    p.add_argument("--gain-smoothing", type=float, default=0.8,
                   help="the weight a running average of each depth's per-step val gains keeps each epoch")
    p.add_argument("--full-half-life", type=float, default=4.0,
                   help="epochs over which a depth's chance of training with explore(n) steps halves "
                        "(--dynamic-steps)")
    p.add_argument("--seeded", action="store_true", help="start each node from its parent's predicted direction")
    p.add_argument("--side-m-history", action="store_true",
                   help="history tokens also read each step's side and m (384's recipe; by default they read only "
                        "each delta, which 452a showed loses nothing)")
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
    p.add_argument("--bf16", action="store_true",
                   help="run the network under bfloat16 autocast (geometry, partitions and losses stay float32)")
    p.add_argument("--grad-steps", type=int, default=0,
                   help="backpropagate through only this many of a node's Lloyd steps, drawn at random per piece "
                        "(every step still runs; the gradient is the same in expectation); 0: all")
    p.add_argument("--grad-tilt", type=float, default=0.1,
                   help="with --grad-steps, step i of M is drawn with probability proportional to 1 + tilt - i / M "
                        "(so the last about tilt / (1 + tilt) as often as the first), each drawn step's loss weighted "
                        "so the gradient is all steps' average in expectation; negative: all steps alike")
    p.add_argument("--grad-prob-min", type=float, default=1.0,
                   help="train each depth on a step with a chance rising linearly from this at depth 1 to 1 at the "
                        "deepest (its loss weighted by 1 / the chance, so the expected gradient is unchanged; the "
                        "other steps only split the batch there); 1: every depth every step")
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
    p.add_argument("--dataset", choices=sorted(SHAPES), default="mnist")
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
        args.side_m_history = not cfg.get("plain_history", False)
        shape = tuple(cfg.get("shape", (1, 28, 28)))
        args.dataset = next(name for name, known in SHAPES.items() if known == shape)
    log(f"=== {time.strftime('%Y-%m-%d')} {vars(args)}")
    X = pixels(load_images(args.dataset, "train", args.data_root)[:args.limit_train]).to(device)
    V = pixels(load_images(args.dataset, "val", args.data_root)[:args.limit_val]).to(device)
    D = X.shape[1]
    power = 20 if D > 1024 else 0  # (start tracking: power iterations for large images, see refresh)
    mean = X.mean(0) if ckpt is None else ckpt["mean"].to(device)
    K, N = args.depths if args.depths > 0 else math.inf, len(X)
    R = X - mean
    root_top = top_direction(R)
    root_gram = R.T @ R / N
    tracking = not args.seeded and args.start == "depth"

    torch.manual_seed(args.seed)
    if ckpt is None:
        encoder = Encoder(1, args.width, args.layers, args.heads, seeded=args.seeded,
                          node_mean=args.node_mean, plain_history=not args.side_m_history,
                          shape=SHAPES[args.dataset]).to(device)
        with torch.no_grad():
            encoder.starts[0] = root_top
            if args.node_mean:
                encoder.mean.copy_(mean)
    else:
        encoder = Encoder(**ckpt["shared"]["config"]).to(device)
        encoder.load_state_dict(ckpt["shared"]["state_dict"])
        assert encoder.depths <= K, f"--depths {K} is below the checkpoint's {encoder.depths}"
    encoder.bf16 = args.bf16
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
    assert not (args.dynamic_steps and args.tol > 0), "--dynamic-steps needs --tol 0"
    grown = 0.0  # when (in epochs) the encoder last grew
    caps = [args.lloyd_steps] * encoder.depths  # each depth's Lloyd steps, part of the tree
    joined = [0.0] * encoder.depths  # when (in epochs) each depth joined
    smoothed: list[list[float]] = []  # each depth's running average of its per-step val gains
    step_cost: list[float] = []  # (threshold schedule) each depth's modeled GPU seconds per epoch for one step
    over_cost: list[float] = []  # and for its overhead
    step_fit: list[float] = []  # the cost models' coefficients (see the threshold schedule), averaged over epochs
    over_fit: list[float] = []
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
        joined = (list(ckpt.get("joined") or []) + [0.0] * encoder.depths)[:encoder.depths]
        smoothed = list(ckpt.get("smoothed") or [])
        step_fit, over_fit = list(ckpt.get("step_fit") or []), list(ckpt.get("over_fit") or [])
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
                        optimizer=opt.state_dict(), tracked=tracked.cpu(), grown=grown, caps=caps, joined=joined,
                        smoothed=smoothed, threshold=tau, threshold_since=since, peak_rate=peak,
                        step_fit=step_fit, over_fit=over_fit,
                        grams=None if grams is None or epoch % 10 else grams.cpu()), args.out)
        # (the Gram matrices only every 10 epochs: at many depths of large images they are gigabytes; a resume
        # without them restarts their running averages)

    def fmt(values, digits=5):
        return " ".join(f"{v:.{digits}f}" for v in values)

    def add_depth():
        """One more depth, from its start tracked so far (refreshed now), and track the next one's."""
        nonlocal tracked, grams
        depth = encoder.depths
        if grams is not None:
            refresh(encoder, grams, tracked, power)
        grow(encoder, opt, depth + 1, tracked[depth:depth + 1])
        caps.append(args.lloyd_steps)
        joined.append(now)
        if depth + 1 < K:
            tracked = torch.cat([tracked, root_top[None]])
            if grams is not None:
                grams = torch.cat([grams, root_gram[None]])

    thresholding = args.step_threshold > 0
    assert not (thresholding and (args.dynamic_steps or args.dynamic_growth)), "--step-threshold replaces those"
    adaptive = args.dynamic_steps or thresholding  # (steps re-chosen each epoch: explore and measure past them)
    # the threshold now, the epoch it was set and its best train mse fall per GPU second since
    tau, since, peak = args.step_threshold, first, 0.0
    if ckpt is not None and thresholding:
        tau, since, peak = ckpt.get("threshold", tau), ckpt.get("threshold_since", first), ckpt.get("peak_rate", 0.0)
    clock = DepthClock(device.type == "cuda") if args.dynamic_steps or args.dynamic_growth or thresholding else None
    grow_next = False  # (--dynamic-growth: add a depth as the next epoch starts)

    def explore(n):
        """The steps a depth that uses n tries, to see whether more would pay: half again as many."""
        return max(n + 1, math.ceil(1.5 * n))

    now = float(first)
    for epoch in range(first, first + args.epochs):
        t0, last = time.time(), time.time()
        ran = [0] * encoder.depths  # Lloyd steps run at each depth, summed over the epoch's batches
        batched = [0] * encoder.depths  # the epoch's batches each depth saw
        backed = [0] * encoder.depths  # Lloyd steps backpropagated at each depth, summed over the batches
        encoder.train()
        totals = torch.zeros(4, encoder.depths, device=device)  # (widened as the encoder grows)
        last_sse = torch.zeros((), device=device)  # after each batch's last token
        gen = torch.Generator().manual_seed(args.seed * 1_000_003 + epoch)
        batches, seen = epoch_batches(N, codes, args, gen), 0
        codes = torch.zeros(N, dtype=torch.int64)
        for i, idx in enumerate(batches):
            now = epoch + i / len(batches)
            scheduled = not (args.dynamic_growth or thresholding) and now - grown >= args.grow_every - 1e-9
            if encoder.depths < K and (scheduled or (grow_next and i == 0)):
                add_depth()
                grown += args.grow_every
            weights = None
            if args.grad_prob_min < 1:  # shallower depths train on fewer steps, their loss weighted up to match
                k = encoder.depths
                odds = [args.grad_prob_min + (1 - args.grad_prob_min) * d / max(k - 1, 1) for d in range(k)]
                draws = torch.rand(k, generator=gen).tolist()
                weights = [1 / q if u < q else None for q, u in zip(odds, draws)]
            run = caps
            if adaptive:  # each depth explores past its steps with its chance, else runs its chosen number
                chance = torch.tensor([0.5 ** ((now - j) / args.full_half_life) for j in joined])
                every = (torch.rand(len(joined), generator=gen) < chance).tolist()
                run = [explore(c) if e else c for e, c in zip(every, caps)]
            ran = (ran + [0] * len(run))[:len(run)]
            batched = (batched + [0] * len(run))[:len(run)]
            backed = (backed + [0] * len(run))[:len(run)]
            for d, n in enumerate(run):
                ran[d] += n
                batched[d] += 1
                backed[d] += n if args.grad_steps <= 0 else min(n, args.grad_steps)
            opt.zero_grad(set_to_none=True)
            stats, sse, keys = train_step(encoder, X[idx.to(device)], mean, grams, args, run, weights, clock)
            last_sse += sse
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
                refresh(encoder, grams, tracked, power)
            if time.time() - last >= args.log_interval:
                last = time.time()
                log.detail(f"    step {step} ({seen}/{N} images this epoch)")
        t1 = time.time()
        parts = clock.totals(encoder.depths) if clock is not None else None
        depth_seconds = None if parts is None else parts["depth"]
        val, val_steps, capped, step_gains = evaluate(encoder, V, mean, args, caps,
                                                      [explore(c) for c in caps] if adaptive else None)
        totals = totals[:, :encoder.depths]  # (a depth added mid-epoch counts only the images it saw)
        objective = (totals[0] / (N * D)).tolist()
        pca = (totals[1] / (N * D)).tolist()
        train_mse = last_sse.item() / (N * D)
        train_steps = (totals[2] / totals[3].clamp_min(1)).tolist()
        full = encoder.depths == K
        improved = full and val[-1] < best[0]
        if improved:
            best = (val[-1], epoch)
        chosen, cost = list(caps), [0.0] * encoder.depths
        # what training buys now: val mse's fall per epoch over the last --rate-window epochs, at the deepest
        # depth the run already had then (so new depths, which arrive whatever a step or a depth costs, don't
        # count), over a horizon of --cost-window epochs
        window = history[-args.rate_window:]
        k = len(window[0]["val_mse"]) if window else 0
        rate = max(window[0]["val_mse"][k - 1] - val[k - 1], 0.0) / len(window) if window else 0.0
        fall = rate * args.cost_window
        train_seconds = max(t1 - t0, 1e-9)
        if adaptive:  # each depth's per-step val gains, averaged over epochs
            b = args.gain_smoothing
            # (the steps measured change length: those measured before average in, new ones start as measured,
            # and ones not measured now keep their last value)
            smoothed = [[b * a + (1 - b) * x for a, x in zip(smoothed[d], g)] + g[len(smoothed[d]):]
                        + smoothed[d][len(g):] if d < len(smoothed) else list(g) for d, g in enumerate(step_gains)]
        grow_next = False
        if thresholding:  # steps: the best cumulative gain less the threshold's price; grow while the deepest's
            assert parts is not None  # first step beats its price
            # each depth's costs per epoch, from its timed parts: a step's forward pass, a backpropagated step's
            # backward pass (the backward less the memory's, about twice its forward), and the depth's overhead
            # (the rest: memory, routing and the like, which no step count changes); averaged over epochs
            # then, to drop the noise, a model of both per batch over depth: a + b N + c N d, with N the depth's
            # nodes per batch and d the depth (the memory a query reads grows with it), fit over all depths with
            # nonnegative coefficients and averaged over epochs; a step's cost does not depend on its index
            # (every step runs the same forward pass and is priced with a backward pass)
            seen = [d for d in range(encoder.depths) if batched[d] > 0]
            feats = [[1.0, float(totals[3][d]) / batched[d], float(totals[3][d]) / batched[d] * (d + 1)] for d in seen]
            ys_step, ys_over = [], []
            for d in seen:
                runs, backs = ran[d] / batched[d], backed[d] / batched[d]
                forward = parts["steps"][d] / max(runs, 1e-9)
                back = max(parts["backward"][d] - 2 * parts["memory"][d], 0.0) / max(backs, 1e-9)
                ys_step.append((forward + back) / batched[d])  # (a step we rely on must also be trained)
                ys_over.append(max(parts["depth"][d] + parts["route"][d] - forward * runs - back * backs, 0.0)
                               / batched[d])
            step_fit = [b * a + (1 - b) * x for a, x in zip(step_fit, nonneg_fit(feats, ys_step))] if step_fit \
                else nonneg_fit(feats, ys_step)
            over_fit = [b * a + (1 - b) * x for a, x in zip(over_fit, nonneg_fit(feats, ys_over))] if over_fit \
                else nonneg_fit(feats, ys_over)
            per_epoch = len(batches)

            def modeled(fit, nodes, depth):  # GPU seconds per epoch
                return per_epoch * (fit[0] + fit[1] * nodes + fit[2] * nodes * (depth + 1))

            nodes_now = [float(totals[3][d]) / max(batched[d], 1) for d in range(encoder.depths)]
            step_cost = [modeled(step_fit, n, d) for d, n in enumerate(nodes_now)]
            over_cost = [modeled(over_fit, n, d) for d, n in enumerate(nodes_now)]
            secs = step_cost
            chosen = []
            for g, c in zip(smoothed, secs):  # (noise in the gains past the useful steps cancels in the sum)
                total = list(itertools.accumulate(g))
                chosen.append(max(range(1, len(total) + 1), key=lambda n: total[n - 1] - tau * c * n))
            # a new depth pays its overhead and a first step, at about twice the deepest depth's nodes
            new_nodes, new_depth = min(2 * nodes_now[-1], float(args.batch)), encoder.depths
            grow_next = encoder.depths < K and smoothed[-1][0] > tau * (modeled(step_fit, new_nodes, new_depth)
                                                                       + modeled(over_fit, new_nodes, new_depth))
            # how fast train mse falls under this threshold: per GPU second over the last --rate-epochs epochs
            now_entry = dict(epoch=epoch, train_mse=train_mse, train_seconds=t1 - t0)
            span = [h for h in history + [now_entry] if h["epoch"] >= since][-(args.rate_epochs + 1):]
            rate = math.nan
            if len(span) == args.rate_epochs + 1:
                seconds = sum(h["train_seconds"] for h in span[1:])
                rate = (span[0]["train_mse"] - span[-1]["train_mse"]) / max(seconds, 1e-9)
                peak = max(peak, rate)
            log(f"    threshold {tau:.2e} val mse per GPU second (a step costs {' '.join(f'{c:.1e}' for c in secs)} s "
                f"per epoch, a depth's overhead {' '.join(f'{c:.1e}' for c in over_cost)} s): "
                f"steps per depth {' '.join(map(str, caps))}, next "
                f"{' '.join(map(str, chosen))}; " + ("grow" if grow_next else "no growth")
                + f"; train mse falling {rate:.2e}/s (best {peak:.2e}/s under this threshold)")
            if not math.isnan(rate) and rate < args.slowdown * peak:
                tau, since, peak = tau / 2, epoch + 1, 0.0
                log(f"    training slowed below {args.slowdown} of its best: threshold halves to {tau:.2e}")
        if args.dynamic_steps:  # next epoch's steps: the best averaged cumulative gain less each step's price
            assert depth_seconds is not None
            for d, g in enumerate(smoothed):  # one step's price: the fall times its share of the training time
                per_step = depth_seconds[d] / max(ran[d] / max(batched[d], 1), 1e-9)
                cost[d] = fall * per_step / train_seconds
                gains = list(itertools.accumulate(g))
                chosen[d] = max(range(1, len(gains) + 1), key=lambda n: gains[n - 1] - cost[d] * n)
        history.append(dict(epoch=epoch, val_mse=val, train_mse=train_mse, objective=objective, pca=pca, train_steps=train_steps,
                            val_steps=val_steps, val_capped=capped, trained=encoder.depths - 1, caps=list(caps),
                            step_gains=step_gains, seconds=time.time() - began, train_seconds=t1 - t0,
                            **(dict(smoothed_gains=smoothed, chosen=chosen, step_cost=cost) if adaptive else {}),
                            **(dict(threshold=tau) if thresholding else {})))
        plot_depths(history, args.plot or os.path.splitext(args.log)[0] + ".png")
        log(f"  epoch {epoch:2d}: val mse after {encoder.depths} tokens {val[-1]:.6f} (psnr {psnr(val[-1]):.2f} dB); "
            "per depth " + fmt(val) + f"; train mse {train_mse:.6f}; objective per depth " + fmt(objective)
            + (("; pca per depth " + fmt(pca)) if args.seeded else "")
            + "; lloyd steps per depth train " + fmt(train_steps, 2) + ", val " + fmt(val_steps, 2)
            + ", val capped " + fmt(capped, 2) + "; nodes per batch " + fmt((totals[3] / len(batches)).tolist(), 0)
            + f" ({t1 - t0:.0f}s train, {time.time() - t1:.0f}s val){'  *' if improved else ''}")
        if args.dynamic_growth and encoder.depths < K:  # the next token's expected gain against its price
            assert depth_seconds is not None
            gain = [a - b for a, b in zip([math.inf] + val[:-1], val)]  # each token's val gain (the first's unknown)
            if len(val) < 3:
                grow_next, expected, price = True, math.nan, math.nan  # (too few tokens to extrapolate)
            else:
                expected = gain[-1] * min(max(gain[-1] / max(gain[-2], 1e-12), 0.0), 1.0)
                price = fall * depth_seconds[-1] / train_seconds
                grow_next = expected > price
            log(f"    next token's expected gain {expected:.2e}, its price {price:.2e}: "
                + ("grow" if grow_next else "wait"))
        if args.dynamic_steps:
            log(f"    steps used per depth {' '.join(map(str, caps))}; next {' '.join(map(str, chosen))}; one step's "
                "price per depth " + " ".join(f"{c:.1e}" for c in cost))
        if not full or improved:
            save(epoch)  # the latest while growing, then the best (with the steps val was routed with)
        elif stagnant(epoch, best[1], args):
            break
        caps[:] = chosen
    if best[1] < 0:
        log(f"stopped at {encoder.depths} of {K} depths; the latest state is in {args.out}")
        return
    log(f"kept epoch {best[1]}: val mse after {K} tokens {best[0]:.6f} (psnr {psnr(best[0]):.2f} dB), saved {args.out}")


if __name__ == "__main__":
    main()
