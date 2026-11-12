"""Stage 4: train the prior (dvae/prior.py's TokenPrior), a causal transformer over the inner encoder's codes that
predicts each token (one of V children), on the codes a train_inner_components.py run keeps in its tree/ folder.

The prior reads each node's children (their centroids) and the tokens before, not the shares the encoder implies
(dvae/codes.py's TreeCodes), and starts uniform; it is trained on cross-entropy alone. Each step takes a batch of
training codes and minimizes the cross-entropy of every token, all depths at once. Every epoch logs, in bits, the
train codes' cost (a fixed subset, scored as val is) and the val codes' cost per image and per range of depths,
beside two baselines the prior has to beat: the encoder's implied shares and the best constant share of each token
per depth (fit on the train codes). Early stopping keeps the epoch with the lowest val bits per image and stops once
a --patience-frac share of the run's epochs (at least --patience) passes without a better one (dvae/stopping.py);
the kept prior is then scored on the test codes against the same baselines.

With --replay (automatic when the run kept no r), each batch's children offsets are computed by replaying the
run's inner networks on its images (dvae/replay.py) instead of read from tree/, so a deep tree needs no stored r.
Every epoch it also samples --samples codes from the prior and as many from the encoder's own shares, decoded.

The progress plot (the log's path as .png, dvae/plots.py's plot_prior), redrawn every epoch: bits per token by code
position (the kept epoch's train, val and test, beside the baselines), bits per image against hours, each epoch's
seconds by part with the GPU's use, and the samples.

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
from dvae.data import SHAPES, get_device
from dvae.logs import RunLog
from dvae.plots import plot_prior
from dvae.components import Utilization
from dvae.prior import TokenPrior, code_token_bits, load_token_prior
from dvae.codes import LazyRows
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
    p.add_argument("--tree", required=True,
                   help="a train_inner_components.py run's folder, whose tree/ holds the codes")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--images", action="store_true",
                   help="also show each token the children's centroids it chooses between, as images (dvae/prior.py)")
    p.add_argument("--wd", type=float, default=0.05, help="AdamW weight decay (on weight matrices only)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500, help="steps of linear learning-rate warmup")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1000)
    add_patience_args(p)
    p.add_argument("--init", default=None, metavar="CKPT",
                   help="start from an earlier prior.pt's weights (one per band in old checkpoints: the latest is taken)")
    p.add_argument("--samples", type=int, default=12, help="codes sampled each epoch, from the prior and from the "
                                                        "encoder's own shares (needs the run's networks)")
    p.add_argument("--train-eval", type=int, default=5000, help="train codes scored each epoch, beside val")
    p.add_argument("--replay", action="store_true",
                   help="compute each batch's children offsets by replaying the run's networks (dvae/replay.py) "
                        "instead of reading r from tree/ (automatic when the run kept none)")
    p.add_argument("--depths", type=int, default=None, help="tokens per code (default: the run's kept depths)")
    p.add_argument("--node-batch", type=int, default=4096, help="nodes per forward pass of the replay")
    p.add_argument("--replay-chunk", type=int, default=512,
                   help="images replayed at once into host memory, then trained on a batch at a time (the replay "
                        "costs about the same per call however many images it holds; rounded to whole batches)")
    p.add_argument("--buffer", type=int, default=10000,
                   help="with the replay: training images whose offsets are kept in host memory (8-bit r, ~K * D bytes "
                        "each) and trained on each epoch, a rolling window over all of them (0: replay every training "
                        "image every epoch)")
    p.add_argument("--rotate", type=int, default=1000, help="images the buffer swaps at the first epoch")
    p.add_argument("--rotate-min", type=int, default=128)
    p.add_argument("--rotate-max", type=int, default=5000)
    p.add_argument("--gap-tol", type=float, default=0.5,
                   help="bits per image: each epoch the oldest images in the buffer and the next ones in (out of it "
                        "longest) are scored before the swap; above this gap the next swap doubles, under half of "
                        "it it halves")
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
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    log = RunLog(args.log, device)
    log(f"=== train_prior.py {vars(args)}")

    began = time.time()
    import json
    with open(os.path.join(args.tree, "records.json")) as f:
        run_args = json.load(f)["args"]
    replay = args.replay or not run_args.get("store_train_r", True) or not run_args.get("store_eval_r", True)
    train, val, test = (TreeCodes(args.tree, s, args.depths, need_r=not replay) for s in ("train", "val", "test"))
    rp = TreeReplay(args.tree, device, train.depths, args.node_batch) if replay else None
    inputs = {s: rp.inputs(s, args.data_root) for s in ("train", "val", "test")} if rp is not None else {}
    if rp is not None:
        log(f"replaying {args.tree}'s {train.depths} depths ({sum(rp.steps)} networks) for every batch")

    def offsets_of(codes: TreeCodes, split: str, rows=None):
        """codes' children offsets, read from tree/ or replayed (rows: which of split's inputs codes holds)."""
        if rp is None:
            return codes.offsets
        X, replayer = (inputs[split] if rows is None else inputs[split][rows]), rp
        return LazyRows(len(X), lambda idx: replayer(X[idx], codes.tokens[idx])[0])  # (along the run's own codes)
    N, K, V = len(train), train.depths, train.V
    log(f"codes: {N} train, {len(val)} val, {len(test)} test, {K} tokens of {V} children, from {train.ckpt} "
        f"({time.time() - began:.0f}s to load)")

    freq = torch.zeros(K, V).scatter_add_(1, train.tokens.T, torch.ones(K, N)) / N  # each token's share per depth

    def baselines(codes: TreeCodes):
        constant = -freq.clamp_min(1e-6).log2().gather(1, codes.tokens.T).T
        return own_bits(codes).mean(0), constant.mean(0)

    chunk = max(1, args.replay_chunk // args.batch) * args.batch

    @torch.no_grad()
    def cost(model, codes: TreeCodes, split: str, rows=None, timing: dict | None = None):
        """The bits each of codes' tokens costs under model, mean over codes [K]; timing (if given) gets the
        replay's seconds added to its "replay"."""
        if rp is None:
            return code_token_bits(model, offsets_of(codes, split, rows), codes.tokens, device, args.batch).mean(0)
        model.eval()  # (as code_token_bits, a chunk replayed at a time)
        X = inputs[split] if rows is None else inputs[split][rows]
        bits = []
        for c in range(0, len(X), chunk):
            t0 = time.time()
            host = rp.to_host(X[c:c + chunk], codes.tokens[c:c + chunk])
            if timing is not None:
                timing["replay"] += time.time() - t0
            for i in range(0, len(host), args.batch):
                tb = codes.tokens[c + i:c + i + args.batch].to(device)
                logits = model(host[i:i + args.batch].to(device, non_blocking=True).float(), tb)
                bits.append(F.cross_entropy(logits.flatten(0, 1).float(), tb.flatten(), reduction="none")
                            .view(tb.shape).cpu() / LN2)
        return torch.cat(bits).mean(0)

    own, constant = baselines(val)
    log(f"val baselines: the encoder's implied shares {describe(own)}")
    log(f"val baselines: constant share per depth {describe(constant)}")

    D = int(train.mean.numel())
    shape = next((sh for sh in SHAPES.values() if sh[0] * sh[1] * sh[2] == D), (1, 28, 28))
    torch.manual_seed(args.seed)  # (the same start with or without a replay, whose networks drew from the generator)
    prior = TokenPrior(K, V, args.width, args.layers, args.heads, args.dropout, D=D, images=args.images,
                       shape=shape).to(device)
    if args.images:  # the children's centroids are offsets from the predicted means, which start from the training mean
        prior.mean.copy_(train.mean.to(device))
    if args.init:
        ckpt = torch.load(args.init, map_location=device, weights_only=False)
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt["states"][max(ckpt["states"])]
        prior.load_state_dict(state)
        log(f"started from {args.init} (its epoch {ckpt.get('epoch')})")
    decay = [q for q in prior.parameters() if q.dim() >= 2]
    rest = [q for q in prior.parameters() if q.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=args.wd), dict(params=rest, weight_decay=0.0)],
                            lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
    log(f"prior: {sum(q.numel() for q in prior.parameters()) / 1e6:.2f}M parameters")

    gen = torch.Generator().manual_seed(args.seed)
    seen = torch.randperm(N, generator=gen)[:args.train_eval]
    train_seen = train.subset(seen)
    if rp is not None:  # (the replay gives the run's own codes: check, on a batch)
        got = rp(inputs["train"][seen[:args.batch]])[1].cpu()
        log(f"replayed codes, left to themselves, match tree/'s on {(got == train_seen.tokens[:args.batch]).float().mean().item():.4%}"
            f" of {got.numel()} tokens (low-precision rounding in other batch shapes); the replay follows tree/'s")
    sampler = rp
    if sampler is None and args.samples > 0 and os.path.isdir(os.path.join(args.tree, "models")):
        sampler = TreeReplay(args.tree, device, K, args.node_batch)
    util = Utilization(device)
    own_tr, constant_tr = baselines(train_seen)
    records: dict = dict(K=K, history=[], best_epoch=None, baselines=dict(
        val_own=own.tolist(), val_constant=constant.tolist(), train_own=own_tr.tolist(),
        train_constant=constant_tr.tolist()))
    samples_path = os.path.splitext(args.plot)[0] + "_samples.pt"
    run_began, step, best = time.time(), 0, (math.inf, -1)
    buffered = rp is not None and args.buffer > 0
    B, ring, cursor, rotate = 0, 0, 0, 0
    store = val_store = CompactStore(0, K, 1)
    cycle = torch.zeros(0, dtype=torch.long)
    if buffered:  # (a rolling window of the training images, their offsets in host memory, and val's)
        assert rp is not None
        B = min(args.buffer, N)
        Dz = rp.D
        log(f"buffer: {B} training images and the {len(val)} val images in host memory "
            f"({CompactStore.gb(B + len(val), K, Dz):.1f} GB)")
        store, val_store = CompactStore(B, K, Dz), CompactStore(len(val), K, Dz)
        cycle = torch.randperm(N, generator=gen)  # (the order the training images pass through the buffer)

        def fill(st: CompactStore, split: str, codes: TreeCodes, ids, slots):
            for c in range(0, len(ids), chunk):
                i, sl = ids[c:c + chunk], slots[c:c + chunk]
                rp.into_store(inputs[split][i], codes.tokens[i], st, sl)
                st.tokens[sl] = codes.tokens[i]

        t0 = time.time()
        fill(store, "train", train, cycle[:B], torch.arange(B))
        fill(val_store, "val", val, torch.arange(len(val)), torch.arange(len(val)))
        log(f"buffer filled in {time.time() - t0:.0f}s")
        ring, cursor, rotate = 0, B % N, min(args.rotate, B)

    @torch.no_grad()
    def bits_of(st: CompactStore, rows) -> torch.Tensor:
        """Each token's bits [len(rows), K] under the prior, for st's rows."""
        prior.eval()
        out = []
        for i in range(0, len(rows), args.batch):
            r = rows[i:i + args.batch]
            tb = st.tokens[r].to(device)
            logits = prior(st.offsets(r, device), tb)
            out.append(F.cross_entropy(logits.flatten(0, 1).float(), tb.flatten(), reduction="none")
                       .view(tb.shape).cpu() / LN2)
        return torch.cat(out)

    for epoch in range(args.epochs):
        began = time.time()
        timing = dict(train=0.0, replay=0.0, eval=0.0, samples=0.0)
        extra: dict = {}
        swap_note = ""
        running, trained_on = 0.0, 0
        if buffered:
            # (the oldest images in the buffer against the next ones in, which have been out of it longest: scored
            # before the swap, their gap says how much the prior has fit the buffer itself)
            t0 = time.time()
            slots = (ring + torch.arange(rotate)) % B
            ids = cycle[(cursor + torch.arange(rotate)) % N]
            old_bits = bits_of(store, slots)
            timing["eval"] += time.time() - t0
            t0 = time.time()
            fill(store, "train", train, ids, slots)
            timing["replay"] += time.time() - t0
            t0 = time.time()
            new_bits = bits_of(store, slots)
            timing["eval"] += time.time() - t0
            gap = new_bits.sum(1).mean().item() - old_bits.sum(1).mean().item()
            extra = dict(fresh=new_bits.mean(0).tolist(), gap=gap, rotate=rotate)
            swap_note = (f"buffer: swapped {rotate} (oldest {old_bits.sum(1).mean().item():.3f}, incoming "
                         f"{new_bits.sum(1).mean().item():.3f} bits/image: gap {gap:+.3f}); ")
            ring, cursor = (ring + rotate) % B, (cursor + rotate) % N
            prior.train()
            for idx in torch.randperm(B, generator=gen).split(args.batch):
                t0 = time.time()
                tokens = store.tokens[idx].to(device, non_blocking=True)
                logits = prior(store.offsets(idx, device), tokens)
                loss = F.cross_entropy(logits.flatten(0, 1), tokens.flatten())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
                opt.step()
                sched.step()
                step += 1
                running += loss.item() * len(idx)
                trained_on += len(idx)
                timing["train"] += time.time() - t0
                if step % 100 == 0:
                    log.detail(f"  step {step}: loss {loss.item() / LN2:.4f} bits/token")
            t0 = time.time()
            tr, va = old_bits.mean(0), bits_of(val_store, torch.arange(len(val))).mean(0)
            timing["eval"] += time.time() - t0
            if epoch > 0:  # (the first swap's oldest images had not been trained on more than the rest)
                rotate = min(rotate * 2, args.rotate_max, B) if gap > args.gap_tol else \
                    max(rotate // 2, args.rotate_min) if gap < args.gap_tol / 2 else rotate
        else:
            prior.train()
            order = torch.randperm(N, generator=gen)
            for i in range(0, N, args.batch):
                idx = order[i:i + args.batch]
                tokens = train.tokens[idx].to(device, non_blocking=True)
                if rp is not None:
                    if i % chunk == 0:  # (the next chunk of this epoch's order, replayed at once)
                        t0 = time.time()
                        cidx = order[i:i + chunk]
                        host = rp.to_host(inputs["train"][cidx], train.tokens[cidx])
                        timing["replay"] += time.time() - t0
                    offsets = host[i % chunk:i % chunk + args.batch].to(device, non_blocking=True)
                else:
                    offsets = train.offsets[idx].to(device, non_blocking=True)
                t0 = time.time()
                logits = prior(offsets.float(), tokens)
                loss = F.cross_entropy(logits.flatten(0, 1), tokens.flatten())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
                opt.step()
                sched.step()
                step += 1
                running += loss.item() * len(idx)
                trained_on += len(idx)
                timing["train"] += time.time() - t0
                if step % 100 == 0:
                    log.detail(f"  step {step}: loss {loss.item() / LN2:.4f} bits/token")
            t0, replayed = time.time(), timing["replay"]
            tr, va = cost(prior, train_seen, "train", seen, timing), cost(prior, val, "val", None, timing)
            timing["eval"] += time.time() - t0 - (timing["replay"] - replayed)  # (scoring, less its replay)
        improved = va.sum().item() < best[0]
        if improved:
            best = (va.sum().item(), epoch)
            records["best_epoch"] = epoch
            torch.save(dict(config=prior.config, state_dict=prior.state_dict(), args=vars(args), codes=args.tree,
                            encoder=train.ckpt, epoch=epoch, history=records["history"]), args.out)
        samples = None
        if sampler is not None and args.samples > 0:
            t0 = time.time()
            prior.eval()
            sg = torch.Generator(device=device).manual_seed(args.seed)  # (the same draws every epoch)
            _, from_prior = sampler.sample(args.samples, prior, sg)
            sg = torch.Generator(device=device).manual_seed(args.seed)
            _, from_own = sampler.sample(args.samples, None, sg)
            samples = dict(prior=from_prior.cpu(), own=from_own.cpu(), shape=sampler.image_shape())
            torch.save(samples, samples_path)
            timing["samples"] += time.time() - t0
        records["history"].append(dict(epoch=epoch, train_loss=running / max(trained_on, 1) / LN2, train=tr.tolist(),
                                       val=va.tolist(), **extra,
                                       seconds=time.time() - run_began, wall=time.time() - began,
                                       timing=timing, util=util.mean(began)))
        plot_prior(records, args.plot, samples)
        stop = stagnant(epoch, best[1], args)
        if swap_note:
            log(swap_note + f"next swap {rotate}")
        log(f"epoch {epoch}: val {describe(va)}; train {describe(tr)}; training loss {running / max(trained_on, 1) / LN2:.4f} "
            f"bits/token; baselines: constant {constant.sum():.3f}, encoder's shares {own.sum():.3f}; best "
            f"{best[0]:.3f} at epoch {best[1]}{' (this one)' if improved else ''}; {time.time() - began:.0f}s: "
            + ", ".join(f"{k} {v:.0f}" for k, v in timing.items())
            + (f"; GPU {records['history'][-1]['util']:.0f}% busy" if records["history"][-1]["util"] is not None
               else ""))
        if stop:
            log(f"{patience(epoch, args)} epochs without a better val cost: stopping")
            break

    kept = load_token_prior(torch.load(args.out, map_location=device, weights_only=False), device)
    own, constant = baselines(test)
    te = cost(kept, test, "test")
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
