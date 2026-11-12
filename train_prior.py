"""Stage 4: train the prior (dvae/prior.py's TokenPrior), a causal transformer over the inner encoder's codes that
predicts each token (one of V children), on the codes a train_inner_components.py run keeps in its tree/ folder.

The prior reads each node's children (their centroids, offsets from the node's mean) and the tokens before, not the
shares the encoder implies, and starts uniform; it is trained on cross-entropy alone, all depths at once. The
children offsets are not stored (r for every image at every depth of a deep tree would take hundreds of GB): they
come from replaying the run's networks along each image's code (dvae/replay.py), into a rolling buffer of --buffer
training images kept in host memory (8-bit r and m), beside val's, replayed once.

Each epoch: --probe images of the buffer (evenly spread by age) and its oldest are scored; the next --rotate training images (cycling through all
of them, so those out of the buffer longest) are replayed into the oldest ones' places and scored before any
training on them. The gap between the incoming and the replaced images says how much the prior has fit the buffer
itself: above --gap-tol bits per image the next swap doubles, under half of it it halves. Then one pass over the
buffer, and val. Every epoch logs, in bits, val beside two baselines the prior has to beat: the encoder's implied
shares and the best constant share of each token per depth (fit on the train codes). Early stopping keeps the epoch
with the lowest val bits per image and stops once a --patience-frac share of the run's epochs (at least --patience)
passes without a better one (dvae/stopping.py); the kept prior is then scored on the test codes.

The progress plot (the log's path as .png, dvae/plots.py's plot_prior), redrawn every epoch: bits per token by code
position, bits per image against hours, bits per image by age in the buffer, each epoch's minutes by part with the
GPU's use, and codes sampled from the prior and from the encoder's own shares.

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
    p.add_argument("--rotate-min", type=int, default=128)
    p.add_argument("--rotate-max", type=int, default=5000)
    p.add_argument("--gap-tol", type=float, default=0.5,
                   help="bits per image between incoming and replaced images above which the next swap doubles (under "
                        "half of it, it halves)")
    p.add_argument("--init", default=None, metavar="CKPT", help="start from an earlier prior.pt's weights")
    p.add_argument("--samples", type=int, default=12,
                   help="codes sampled each epoch, from the prior and from the encoder's own shares")
    p.add_argument("--depths", type=int, default=None, help="tokens per code (default: the run's kept depths)")
    p.add_argument("--node-batch", type=int, default=4096, help="nodes per forward pass of the replay")
    p.add_argument("--replay-chunk", type=int, default=512, help="images replayed at once (it costs about the same "
                                                                   "per call however many it holds)")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tf32", action="store_true")
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
        for offsets, tb in st.batches(list(torch.as_tensor(rows).split(args.batch)), device):
            logits = prior(offsets, tb)
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
    run_began, step, best = time.time(), 0, (math.inf, -1)
    for epoch in range(args.epochs):
        began = time.time()
        timing = dict(train=0.0, replay=0.0, eval=0.0, samples=0.0)
        # (a probe of the buffer evenly spread by age, and its oldest images; then the next images in, which have been
        # out of it longest, replayed into the oldest ones' places and scored before any training on them)
        t0 = time.time()
        by_age = age.argsort()
        ranks = torch.linspace(0, B - 1, min(args.probe, B)).round().long()
        probe_bits = bits_of(store, by_age[ranks]).sum(1)
        k = min(rotate, args.probe)
        old_bits = bits_of(store, by_age[:k])
        slots = by_age[:rotate]
        ids = cycle[(cursor + torch.arange(rotate)) % N]
        timing["eval"] += time.time() - t0
        t0 = time.time()
        fill(store, "train", train, ids, slots)
        timing["replay"] += time.time() - t0
        t0 = time.time()
        new_bits = bits_of(store, slots[:k])
        gap = new_bits.sum(1).mean().item() - old_bits.sum(1).mean().item()
        age[slots] = inserted + torch.arange(rotate)
        inserted, cursor, swapped = inserted + rotate, (cursor + rotate) % N, rotate
        timing["eval"] += time.time() - t0

        prior.train()
        running = 0.0
        t0 = time.time()
        for offsets, tokens in store.batches(list(torch.randperm(B, generator=gen).split(args.batch)), device):
            logits = prior(offsets, tokens)
            loss = F.cross_entropy(logits.flatten(0, 1), tokens.flatten())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            running += loss.item() * len(tokens)
            if step % 100 == 0:
                log.detail(f"  step {step}: loss {loss.item() / LN2:.4f} bits/token")
        timing["train"] += time.time() - t0

        t0 = time.time()
        va = bits_of(val_store, torch.arange(len(val))).mean(0)
        timing["eval"] += time.time() - t0
        if epoch > 0:  # (the first swap's oldest images had been trained on no more than the rest)
            rotate = min(rotate * 2, args.rotate_max, B) if gap > args.gap_tol else \
                max(rotate // 2, args.rotate_min) if gap < args.gap_tol / 2 else rotate
        improved = va.sum().item() < best[0]
        if improved:
            best = (va.sum().item(), epoch)
            records["best_epoch"] = epoch
            torch.save(dict(config=prior.config, state_dict=prior.state_dict(), args=vars(args), codes=args.tree,
                            epoch=epoch, history=records["history"]), args.out)
        samples = None
        if args.samples > 0:
            t0 = time.time()
            prior.eval()
            draws = [torch.Generator(device=device).manual_seed(args.seed) for _ in range(2)]  # (the same each epoch)
            samples = dict(prior=rp.sample(args.samples, prior, draws[0])[1].cpu(),
                           own=rp.sample(args.samples, None, draws[1])[1].cpu(), shape=rp.image_shape())
            torch.save(samples, samples_path)
            timing["samples"] += time.time() - t0
        records["history"].append(dict(
            epoch=epoch, train_loss=running / B / LN2, val=va.tolist(), oldest=old_bits.mean(0).tolist(),
            incoming=new_bits.mean(0).tolist(), by_age=dict(x=ranks.tolist(), y=probe_bits.tolist()), gap=gap, swapped=swapped,
            seconds=time.time() - run_began, wall=time.time() - began, timing=timing, util=util.mean(began)))
        plot_prior(records, args.plot, samples)
        log(f"epoch {epoch}: val {describe(va)}; buffer {probe_bits.mean():.3f} (oldest "
            f"{old_bits.sum(1).mean():.3f}, incoming {new_bits.sum(1).mean():.3f}: gap {gap:+.3f}; swapped {swapped}, "
            f"next {rotate}); training loss {running / B / LN2:.4f} bits/token; baselines: constant "
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
