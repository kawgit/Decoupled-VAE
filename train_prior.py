"""Stage 4: train the prior (dvae/prior.py's TokenPrior), a causal transformer over the inner encoder's codes that
predicts each token (one of V children), on the codes a train_inner_components.py run keeps in its tree/ folder.

The prior reads each node's split (the rare child's offset r from the node's mean, and m: the common child's is
-m r) and the tokens before, not the shares the encoder implies, and starts uniform; it is trained on cross-entropy alone, all depths at once. The
children offsets are not stored (r for every image at every depth of a deep tree would take hundreds of GB): they
come from replaying the run's networks along each image's code (dvae/replay.py), into a rolling buffer of --buffer
training images kept in host memory (8-bit r and m), beside val's, replayed once.

Each epoch: the buffer's oldest images are scored (and every --probe-every epochs, --probe images of it evenly
spread by age, for the plot); the next --rotate training images (cycling through all
of them, so those out of the buffer longest) are replayed into the oldest ones' places and scored before any
training on them. The gap between the incoming images and the buffer's oldest (--probe of each at most) says how
much the prior has fit the buffer itself: its running average (--gap-smooth: one image's bits vary by several, so a
swap of a hundred images gives a gap a bit off either way) above --gap-tol bits per image doubles the next swap,
under half of it halves it. Then one pass over the
buffer, and val. The incoming images are replayed a few depths after each step of that pass (on a CUDA stream of
their own), which trains on the rest of the buffer: the images they replace are not trained on again, and the
replay's work, many small kernels, fills the gaps in the training's. Every epoch logs, in bits, val beside two baselines the prior has to beat: the encoder's implied
shares and the best constant share of each token per depth (fit on the train codes). Early stopping keeps the epoch
with the lowest val bits per image and stops once a --patience-frac share of the run's epochs (at least --patience)
passes without a better one (dvae/stopping.py); the kept prior is then scored on the test codes.

The progress plot (the log's path as .png, dvae/plots.py's plot_prior), redrawn every epoch: bits per token by code
position, bits per image against hours, bits per image by age in the buffer, each epoch's minutes by part with the
GPU's use, and codes sampled from the prior (every --samples-every epochs) and from the encoder's own shares (once:
they do not change).

Run from the repo: python train_prior.py --tree artifacts/<inner job>
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F

from dvae.artifacts import artifact
from dvae.codes import TreeCodes, own_bits
from dvae.components import Utilization
from dvae.data import SHAPES, get_device
from dvae.logs import RunLog
from dvae.plots import plot_prior
from dvae.prior import TokenPrior, load_token_prior
from dvae.replay import CompactStore, TreeReplay
from dvae.stopping import add_patience_args, patience, stagnant

LN2 = math.log(2)


def bands(K):
    spans = [(1, 8)] + [(2 ** i + 1, 2 ** (i + 1)) for i in range(3, 20)]
    return [(lo, min(hi, K)) for lo, hi in spans if lo <= K]


def describe(per_token):
    """Bits per image, then bits per token over each range of depths."""
    return f"{per_token.sum():.3f} bits/image (per token " + ", ".join(
        f"{lo}-{hi}: {per_token[lo - 1:hi].mean():.4f}" for lo, hi in bands(len(per_token))) + ")"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tree", required=True, help="a train_inner_components.py run's folder (its tree/ and models/)")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--images", action="store_true",
                   help="also show each token the children's centroids it chooses between, as images (dvae/prior.py)")
    p.add_argument("--wd", type=float, default=0.05, help="AdamW weight decay (on weight matrices only)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500, help="steps of linear learning-rate warmup")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100000)
    add_patience_args(p)
    p.add_argument("--buffer", type=int, default=10000,
                   help="training images whose offsets are kept in host memory (~K * D bytes each)")
    p.add_argument("--rotate", type=int, default=1000, help="images the buffer swaps at the first epoch")
    p.add_argument("--probe", type=int, default=1000,
                   help="buffer images scored each epoch (evenly spread by age), and the most of the oldest and of "
                        "the incoming ones scored for the gap")
    p.add_argument("--probe-every", type=int, default=10, help="epochs between probes of the buffer by age")
    p.add_argument("--gap-smooth", type=float, default=0.3, help="weight of each epoch's gap in its running average")
    p.add_argument("--rotate-min", type=int, default=128)
    p.add_argument("--rotate-max", type=int, default=5000)
    p.add_argument("--gap-tol", type=float, default=0.5,
                   help="bits per image between incoming and replaced images above which the next swap doubles (under "
                        "half of it, it halves)")
    p.add_argument("--init", default=None, metavar="CKPT", help="start from an earlier prior.pt's weights")
    p.add_argument("--samples", type=int, default=12,
                   help="codes sampled from the prior and from the encoder's own shares, for the plot")
    p.add_argument("--samples-every", type=int, default=10, help="epochs between draws from the prior")
    p.add_argument("--depths", type=int, default=None, help="tokens per code (default: the run's kept depths)")
    p.add_argument("--node-batch", type=int, default=4096, help="nodes per forward pass of the replay")
    p.add_argument("--replay-chunk", type=int, default=1024, help="images replayed at once (it costs about the same "
                                                                   "per call however many it holds)")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--bf16", action="store_true", help="the prior computes in bfloat16 (autocast)")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None, help="checkpoint (default: the run's artifacts folder's prior.pt)")
    p.add_argument("--log", default=None, help="log (default: the run's artifacts folder's run.log)")
    p.add_argument("--plot", default=None, help="progress plot, redrawn every epoch (default: --log's path as .png)")
    args = p.parse_args()
    args.out, args.log = args.out or artifact("prior.pt"), args.log or artifact("run.log")
    args.plot = args.plot or os.path.splitext(args.log)[0] + ".png"
    device = get_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    log = RunLog(args.log, device)
    log(f"=== train_prior.py {vars(args)}")

    began = time.time()
    train, val, test = (TreeCodes(args.tree, s, args.depths) for s in ("train", "val", "test"))
    N, K, V = len(train), train.depths, train.V
    rp = TreeReplay(args.tree, device, K, args.node_batch)
    inputs = {s: rp.inputs(s, args.data_root) for s in ("train", "val", "test")}
    log(f"codes: {N} train, {len(val)} val, {len(test)} test, {K} tokens of {V} children; replaying {args.tree}'s "
        f"{sum(rp.steps)} networks ({time.time() - began:.0f}s to load)")

    freq = torch.zeros(K, V).scatter_add_(1, train.tokens.T, torch.ones(K, N)) / N  # each token's share per depth

    def baselines(codes: TreeCodes):
        constant = -freq.clamp_min(1e-6).log2().gather(1, codes.tokens.T).T
        return own_bits(codes).mean(0), constant.mean(0)

    own, constant = baselines(val)
    log(f"val baselines: the encoder's implied shares {describe(own)}")
    log(f"val baselines: constant share per depth {describe(constant)}")

    torch.manual_seed(args.seed)
    D = rp.D
    shape = next((sh for sh in SHAPES.values() if sh[0] * sh[1] * sh[2] == D), (1, 28, 28))
    prior = TokenPrior(K, V, args.width, args.layers, args.heads, args.dropout, D=D, images=args.images,
                       shape=shape).to(device)
    if args.images:  # the children's centroids are offsets from the predicted means, which start from the training mean
        prior.mean.copy_(rp.mean)
    if args.init:
        prior.load_state_dict(load_token_prior(torch.load(args.init, map_location=device, weights_only=False),
                                               device).state_dict())
        log(f"started from {args.init}")
    decay = [q for q in prior.parameters() if q.dim() >= 2]
    rest = [q for q in prior.parameters() if q.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=args.wd), dict(params=rest, weight_decay=0.0)],
                            lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
    log(f"prior: {sum(q.numel() for q in prior.parameters()) / 1e6:.2f}M parameters")
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    def autocast():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16)

    def fill(st: CompactStore, split: str, codes: TreeCodes, ids, slots):
        for c in range(0, len(ids), args.replay_chunk):
            i, sl = ids[c:c + args.replay_chunk], slots[c:c + args.replay_chunk]
            rp.into_store(inputs[split][i], codes.tokens[i], st, sl)
            st.tokens[sl] = codes.tokens[i]

    @torch.no_grad()
    def bits_of(st: CompactStore, rows) -> torch.Tensor:
        """Each token's bits [len(rows), K] under the prior, for st's rows."""
        prior.eval()
        out = []
        for r, m, tb in st.batches(list(torch.as_tensor(rows).split(args.batch)), device, dtype):
            with autocast():
                logits = prior(r, m, tb)
            out.append(F.cross_entropy(logits.flatten(0, 1).float(), tb.flatten(), reduction="none")
                       .view(tb.shape).cpu() / LN2)
        return torch.cat(out)

    gen = torch.Generator().manual_seed(args.seed)
    B = min(args.buffer, N)
    log(f"buffer: {B} training images and the {len(val)} val images in host memory "
        f"({CompactStore.gb(B + len(val), K, D):.1f} GB)")
    store, val_store = CompactStore(B, K, D), CompactStore(len(val), K, D)
    cycle = torch.randperm(N, generator=gen)  # (the order the training images pass through the buffer)
    t0 = time.time()
    fill(store, "train", train, cycle[:B], torch.arange(B))
    fill(val_store, "val", val, torch.arange(len(val)), torch.arange(len(val)))
    log(f"buffer filled in {time.time() - t0:.0f}s")
    age = torch.arange(B)  # (each slot's insertion count: smaller is older)
    inserted, cursor, rotate = B, B % N, min(args.rotate, B)

    util = Utilization(device)
    records: dict = dict(K=K, B=B, history=[], best_epoch=None, baselines=dict(
        val_own=own.tolist(), val_constant=constant.tolist()))
    samples_path = os.path.splitext(args.plot)[0] + "_samples.pt"
    samples: dict | None = None
    if args.samples > 0:  # (the encoder's own: the same every epoch)
        own_samples = rp.sample(args.samples, None, torch.Generator(device=device).manual_seed(args.seed))[1].cpu()
    side = torch.cuda.Stream(device) if device.type == "cuda" else None
    run_began, step, best = time.time(), 0, (math.inf, -1)
    smooth: float | None = None  # (the gap's running average)
    for epoch in range(args.epochs):
        began = time.time()
        timing = dict(train=0.0, replay=0.0, eval=0.0, samples=0.0)
        # (a probe of the buffer evenly spread by age, and its oldest images; then the next images in, which have been
        # out of it longest, replayed into the oldest ones' places and scored before any training on them)
        t0 = time.time()
        by_age = age.argsort()
        probed = None
        if epoch % args.probe_every == 0:
            ranks = torch.linspace(0, B - 1, min(args.probe, B)).round().long()
            probed = dict(x=ranks.tolist(), y=bits_of(store, by_age[ranks]).sum(1).tolist())
        k = min(rotate, args.probe)
        old_bits = bits_of(store, by_age[:min(args.probe, B)])
        slots = by_age[:rotate]
        ids = cycle[(cursor + torch.arange(rotate)) % N]
        timing["eval"] += time.time() - t0
        # (the incoming images replayed into the oldest ones' places a few depths after each training step, on a
        # stream of their own, while the rest of the buffer trains)
        def replay_steps():
            for c in range(0, rotate, args.replay_chunk):
                i, sl = ids[c:c + args.replay_chunk], slots[c:c + args.replay_chunk]
                yield from rp.into_store_steps(inputs["train"][i], train.tokens[i], store, sl, side)
                store.tokens[sl] = train.tokens[i]

        replaying = replay_steps()
        keep = torch.ones(B, dtype=torch.bool)
        keep[slots] = False
        rest = keep.nonzero()[:, 0]
        order = list(rest[torch.randperm(len(rest), generator=gen)].split(args.batch))
        pending = math.ceil(rotate / args.replay_chunk) * (K + 1)  # (the replay's steps left)
        prior.train()
        running, seen = torch.zeros((), device=device), 0  # (summed on the device: no wait on it each step)
        t0 = time.time()
        for done, (r, m, tokens) in enumerate(store.batches(order, device, dtype)):
            with autocast():
                logits = prior(r, m, tokens)
            loss = F.cross_entropy(logits.flatten(0, 1).float(), tokens.flatten())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            running += loss.detach() * len(tokens)
            seen += len(tokens)
            for _ in range(math.ceil(pending / (len(order) - done))):  # (spread evenly over the steps)
                next(replaying, None)
                pending -= 1
            if step % 100 == 0:
                log.detail(f"  step {step}: loss {loss.item() / LN2:.4f} bits/token")
        train_loss = running.item() / seen / LN2
        timing["train"] += time.time() - t0
        t0 = time.time()
        for _ in replaying:  # (whatever is left)
            pass
        timing["replay"] += time.time() - t0

        t0 = time.time()
        new_bits = bits_of(store, slots[:k])
        gap = new_bits.sum(1).mean().item() - old_bits.sum(1).mean().item()
        if epoch > 0:  # (the first swap's oldest images had been trained on no more than the rest)
            smooth = gap if smooth is None else (1 - args.gap_smooth) * smooth + args.gap_smooth * gap
        age[slots] = inserted + torch.arange(rotate)
        inserted, cursor, swapped = inserted + rotate, (cursor + rotate) % N, rotate
        timing["eval"] += time.time() - t0

        t0 = time.time()
        va = bits_of(val_store, torch.arange(len(val))).mean(0)
        timing["eval"] += time.time() - t0
        if smooth is not None:
            rotate = min(rotate * 2, args.rotate_max, B) if smooth > args.gap_tol else \
                max(rotate // 2, args.rotate_min) if smooth < args.gap_tol / 2 else rotate
        improved = va.sum().item() < best[0]
        if improved:
            best = (va.sum().item(), epoch)
            records["best_epoch"] = epoch
            torch.save(dict(config=prior.config, state_dict=prior.state_dict(), args=vars(args), codes=args.tree,
                            epoch=epoch, history=records["history"]), args.out)
        if args.samples > 0 and epoch % args.samples_every == 0:
            t0 = time.time()
            prior.eval()
            draw = torch.Generator(device=device).manual_seed(args.seed)  # (the same each time)
            samples = dict(prior=rp.sample(args.samples, prior, draw, args.bf16)[1].cpu(), own=own_samples,
                           shape=rp.image_shape(), epoch=epoch)
            torch.save(samples, samples_path)
            timing["samples"] += time.time() - t0
        records["history"].append(dict(
            epoch=epoch, train_loss=train_loss, val=va.tolist(), oldest=old_bits.mean(0).tolist(),
            incoming=new_bits.mean(0).tolist(), by_age=probed, gap=gap, gap_smooth=smooth, swapped=swapped,
            seconds=time.time() - run_began, wall=time.time() - began, timing=timing, util=util.mean(began)))
        plot_prior(records, args.plot, samples)
        log(f"epoch {epoch}: val {describe(va)}; buffer{f' {sum(probed["y"]) / len(probed["y"]):.3f}' if probed else ''} (oldest "
            f"{old_bits.sum(1).mean():.3f}, incoming {new_bits.sum(1).mean():.3f}: gap {gap:+.3f}, averaged {smooth if smooth is not None else gap:+.3f}; swapped {swapped}, "
            f"next {rotate}); training loss {train_loss:.4f} bits/token; baselines: constant "
            f"{constant.sum():.3f}, encoder's shares {own.sum():.3f}; best {best[0]:.3f} at epoch {best[1]}"
            f"{' (this one)' if improved else ''}; {time.time() - began:.0f}s: "
            + ", ".join(f"{k} {v:.0f}" for k, v in timing.items())
            + (f"; GPU {records['history'][-1]['util']:.0f}% busy" if records["history"][-1]["util"] is not None
               else ""))
        if stagnant(epoch, best[1], args):
            log(f"{patience(epoch, args)} epochs without a better val cost: stopping")
            break

    kept = load_token_prior(torch.load(args.out, map_location=device, weights_only=False), device)
    prior.load_state_dict(kept.state_dict())
    test_bits = []
    for c in range(0, len(test), args.replay_chunk):  # (a chunk replayed and scored at a time)
        ids = torch.arange(c, min(c + args.replay_chunk, len(test)))
        st = CompactStore(len(ids), K, D)
        fill(st, "test", test, ids, torch.arange(len(ids)))
        test_bits.append(bits_of(st, torch.arange(len(ids))))
    te = torch.cat(test_bits).mean(0)
    own, constant = baselines(test)
    records["test"] = te.tolist()
    records["baselines"].update(test_own=own.tolist(), test_constant=constant.tolist())
    plot_prior(records, args.plot, torch.load(samples_path, weights_only=False) if os.path.exists(samples_path)
               else None)
    log(f"kept epoch {best[1]} (val {best[0]:.3f} bits/image), saved {args.out}")
    log(f"test: prior {describe(te)}")
    log(f"test: encoder's implied shares {describe(own)}")
    log(f"test: constant share per depth {describe(constant)}")


if __name__ == "__main__":
    main()
