"""Train a delta tree (see dvae/deltatree.py): a fresh decoder per depth, each on the frontier its predecessors
built.

At each depth, every frontier node runs Lloyd's algorithm on the decoder from the depth's start, the training
remainders' top principal direction there. At every step the decoder predicts both sides' mean deltas and is
trained on the actual means of the node's training remainders, each side weighted by its image count (so every
image weighs the same), and the next hyperplane is the bisector of the predictions, detached. Held-out images
go down their own tree built the same way, and early stopping watches their mse after the last step's split.
The objective itself is no guide there: it is measured along the decoder's own Lloyd paths, which move to
better but harder-to-predict hyperplanes as it learns. The best decoder then makes that split at every node,
and the side's predicted delta extends each child's history.

With --num-folds F > 0 the training images are split into F folds, and each depth trains F + 1 decoders in
lockstep, from the same initial weights and in the same batch order: fold model i on every fold but the ith,
and the final model on all of them. They differ only in their data, and all keep the epoch with the best
out-of-fold mse after the split (each fold model on its own held-out fold, pooled over every training image);
val is only reported. Training images then go down their own fold's models, so every one's history comes from
models that never saw it, like a held-out image's: a node is a (fold, code) pair, and the folds share their
depth's start. The final models are the tree that val, test and decoding use; the fold models are
kept beside the checkpoint (*_folds.pt, half precision) only so --resume can rebuild the training frontier.

--resume deepens a trained tree: it replays the checkpoint's decoders to rebuild both frontiers, then trains
the depths after them up to --max-depth. The settings that define the tree (the data, the root mean, the folds
and the Lloyd steps) come from the checkpoint; the training settings are the ones given now. Each depth
draws its batches from its own seed, so a resumed run trains the same as one that never stopped.
"""

import argparse
import copy
import math
import os
import time
from typing import Any

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")  # ~6.5 GB: fail instead of swapping
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")

import torch  # noqa: E402

from dvae.data import load_mnist  # noqa: E402
from dvae.deltatree import (D, DeltaDecoder, Frontier, chunk_sse, descend, node_order, pixels,  # noqa: E402
                            root_frontier, top_direction)


TREE_ARGS = ("lloyd_steps", "limit_train", "limit_val", "data_root", "num_folds")


def get_device(name):
    if name:
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train_chunks(fr, size, gen, keep=None):
    """An epoch's batches of the images keep marks (default: all): images grouped by node, nodes in random order,
    images in random order within a node, cut every size images (a node may straddle two batches; its gradient
    still sums exactly)."""
    perm = torch.randperm(len(fr.R), generator=gen)
    if keep is not None:
        perm = perm[keep.cpu()[perm]]
    rank = torch.randperm(len(fr.bits), generator=gen)[fr.node.cpu()[perm]]
    perm = perm[torch.argsort(rank, stable=True)].to(fr.R.device)
    return [perm[i:i + size] for i in range(0, len(perm), size)]


def by_nodes(imgs, node, cap):
    """Cut images grouped by node (imgs, their nodes node) into pieces holding at most cap nodes each: the
    decoder's memory use grows with the nodes it queries, not the images."""
    new = torch.ones_like(node, dtype=torch.long)
    new[1:] = (node[1:] != node[:-1]).long()
    piece = (new.cumsum(0) - 1) // cap
    return list(imgs.split(torch.unique_consecutive(piece, return_counts=True)[1].tolist()))


@torch.no_grad()
def evaluate(decoder, fr, steps, size, cap, keep=None):
    """Over the images keep marks (default: all), the objective per image and pixel (averaged over steps), and
    the last step's split: mse before and after (per pixel) and the token's bits per image."""
    decoder.eval()
    order, _ = node_order(fr)
    if keep is not None:
        order = order[keep[order]]
    total = dict(sse=0.0, before=0.0, after=0.0, bits=0.0)
    for i in range(0, len(order), size):
        for imgs in by_nodes(order[i:i + size], fr.node[order[i:i + size]], cap):
            sse, stats = chunk_sse(decoder, fr, imgs, steps, select=True)
            assert stats is not None
            total["sse"] += sse.item()
            for key, value in stats.items():
                total[key] += value.item()
    N = len(order)
    return dict(loss=total["sse"] / (N * steps * D), mse_before=total["before"] / (N * D),
                mse_after=total["after"] / (N * D), bits=total["bits"] / N)


def psnr(mse):
    return -10 * math.log10(mse)


def train_epoch(decoder, opt, sched, fr, keep, gen, args, log):
    """One epoch of decoder on the images of frontier fr that keep marks (None: all). Returns the objective's
    sum over them."""
    steps, norm = args.lloyd_steps, args.lloyd_steps * D
    decoder.train()
    total, last = fr.R.new_zeros(()), time.time()
    batches = train_chunks(fr, args.batch, gen, keep)
    for step, imgs in enumerate(batches):
        opt.zero_grad(set_to_none=True)
        loss = fr.R.new_zeros(())
        for piece in by_nodes(imgs, fr.node[imgs], args.max_nodes):
            sse, _ = chunk_sse(decoder, fr, piece, steps)
            part = sse / (args.batch * norm)  # a fixed divisor, so every image weighs the same
            part.backward()
            loss += part.detach()
            total += sse.detach()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        opt.step()
        sched.step()
        if time.time() - last >= args.log_interval:
            last = time.time()
            log(f"    step {step + 1}/{len(batches)} loss {loss.item():.6f}")
    return total.item()


def fit_depth(decoders, keeps, fr, held_out, report, args, depth, log, tag):
    """Train decoders in lockstep, decoder i on the images of frontier fr that keeps[i] marks (None: all), one
    epoch each per round, all from the same initial weights (as given) and the same batch order, so they differ
    only in their data. Every round ends with held_out(), whose mse after the split picks the epoch all of them
    keep, and report() (logged only). Returns the kept epoch and its held_out() and report() metrics."""
    norm = args.lloyd_steps * D
    opts = [torch.optim.AdamW(d.parameters(), lr=args.lr, weight_decay=args.weight_decay) for d in decoders]
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, lambda s: min(1.0, (s + 1) / args.warmup_steps)) for o in opts]
    gen_states = [torch.Generator().manual_seed(args.seed * 1000 + depth).get_state()] * len(decoders)
    N = sum(len(fr.R) if k is None else int(k.sum()) for k in keeps)
    best: tuple[float, int, list | None, dict | None, dict | None] = (math.inf, -1, None, None, None)
    for epoch in range(args.epochs):
        t0, total = time.time(), 0.0
        for i, (decoder, opt, sched, keep) in enumerate(zip(decoders, opts, scheds, keeps)):
            gen = torch.Generator()
            gen.set_state(gen_states[i])
            total += train_epoch(decoder, opt, sched, fr, keep, gen, args, log)
            gen_states[i] = gen.get_state()
        t1 = time.time()
        val, rep = held_out(), report()
        improved = val["mse_after"] < best[0]
        log(f"  {tag} epoch {epoch:2d}: {val['name']} mse {val['mse_before']:.6f} -> {val['mse_after']:.6f} "
            f"(psnr {psnr(val['mse_after']):.2f} dB), {val['bits']:.3f} bits"
            + (f"; {rep['name']} mse {rep['mse_after']:.6f}" if rep else "")
            + f"; objective train {total / (N * norm):.6f}, held out {val['loss']:.6f} "
            f"({t1 - t0:.0f}s train, {time.time() - t1:.0f}s eval){'  *' if improved else ''}")
        if improved:
            states = [{key: v.detach().clone() for key, v in d.state_dict().items()} for d in decoders]
            best = (val["mse_after"], epoch, states, val, rep)
        elif epoch - best[1] >= args.patience:
            break
    assert best[2] is not None and best[3] is not None
    for decoder, state in zip(decoders, best[2]):
        decoder.load_state_dict(state)
        decoder.eval()
    return best[1], best[3], best[4]


def combined(metrics):
    """Per-image metrics of several disjoint sets of images ([(count, metrics)]), pooled."""
    N = sum(n for n, _ in metrics)
    return {key: sum(n * m[key] for n, m in metrics) / N for key in metrics[0][1]}


def load_decoder(cfg, state, device):
    decoder = DeltaDecoder(**cfg).to(device)
    decoder.load_state_dict(state)
    return decoder.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-depth", type=int, default=32, help="tokens per code")
    p.add_argument("--num-folds", type=int, default=5,
                   help="route each training image with models that never saw it (0: with the final models)")
    p.add_argument("--lloyd-steps", type=int, default=8)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--batch", type=int, default=1024, help="training images per step")
    p.add_argument("--max-nodes", type=int, default=256,
                   help="nodes per forward pass; a batch spanning more is run in pieces with accumulated gradients")
    p.add_argument("--node-batch", type=int, default=512, help="nodes split at once when descending")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--epochs", type=int, default=30, help="max epochs per depth")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--out", default="checkpoints/deltatree.pt")
    p.add_argument("--resume", nargs="?", const="", default=None, metavar="CKPT",
                   help="deepen CKPT (default: --out): replay its decoders, then train the depths after them")
    p.add_argument("--log", default="outputs/deltatree.log")
    p.add_argument("--log-interval", type=float, default=10.0, help="seconds between step logs")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="data")
    args = p.parse_args()

    device = get_device(args.device)
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    logf = open(args.log, "a")

    def log(msg):
        msg = f"{time.strftime('%H:%M:%S')} {msg}"
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    ckpt = None
    if args.resume is not None:
        ckpt = torch.load(args.resume or args.out, map_location=device, weights_only=False)
        for key in TREE_ARGS:
            built = ckpt["args"][key]
            if getattr(args, key) != built:
                print(f"--{key.replace('_', '-')} {getattr(args, key)} ignored: the tree was built with {built}")
            setattr(args, key, built)
    log(f"=== {time.strftime('%Y-%m-%d')} {vars(args)}")
    X = {"train": pixels(load_mnist("train", args.data_root)[:args.limit_train]).to(device),
         "val": pixels(load_mnist("val", args.data_root)[:args.limit_val]).to(device)}
    mean = X["train"].mean(0) if ckpt is None else ckpt["mean"].to(device)
    n_folds = args.num_folds
    folds = torch.randperm(len(X["train"]), generator=torch.Generator().manual_seed(args.seed)) % n_folds if n_folds else None
    fr: dict[str, Frontier] = {"train": root_frontier(X["train"], mean, folds, n_folds), "val": root_frontier(X["val"], mean)}
    steps = args.lloyd_steps
    folds_path = os.path.splitext(args.out)[0] + "_folds.pt"

    def descend_all(final, fold_decoders):
        fr["train"] = descend(final, fr["train"], steps, args.node_batch, fold_decoders)
        fr["val"] = descend(final, fr["val"], steps, args.node_batch)

    saved: dict[str, Any] = dict(args=vars(args), mean=mean.cpu(), configs=[], decoders=[], metrics=[])
    fold_saved: dict[str, Any] = dict(configs=[], decoders=[])
    if ckpt is not None:
        source = args.resume or args.out
        if n_folds:
            fold_saved = torch.load(os.path.splitext(source)[0] + "_folds.pt", map_location=device, weights_only=False)
        for depth, (cfg, state) in enumerate(zip(ckpt["configs"], ckpt["decoders"])):
            fold_decoders = [load_decoder(c, st, device) for c, st in
                             zip(fold_saved["configs"][depth], fold_saved["decoders"][depth])] if n_folds else None
            descend_all(load_decoder(cfg, state, device), fold_decoders)
            mse = fr["val"].R.square().mean().item()
            log(f"replayed depth {depth}: val mse {mse:.6f} (psnr {psnr(mse):.2f} dB) after {depth + 1} tokens, "
                f"train mse {fr['train'].R.square().mean().item():.6f}")
        saved.update(configs=ckpt["configs"], decoders=ckpt["decoders"], metrics=ckpt["metrics"])
        log(f"resumed {len(ckpt['decoders'])} depths from {source}, saving to {args.out}")

    for depth in range(len(saved["decoders"]), args.max_depth):
        t_depth = time.time()
        scale = fr["train"].R.square().mean().sqrt().item()
        image_fold = None if fr["train"].fold is None else fr["train"].fold.to(device)[fr["train"].node]
        log(f"depth {depth}: {len(fr['train'].bits)} train nodes, {len(fr['val'].bits)} val nodes, "
            f"remainder rms {scale:.4f}, {n_folds + 1 if n_folds else 1} models")
        torch.manual_seed(args.seed + depth)
        decoder = DeltaDecoder(depth, scale, args.width, args.layers, args.heads).to(device)
        with torch.no_grad():
            decoder.start.copy_(top_direction(fr["train"].R))
        fold_models = [copy.deepcopy(decoder) for _ in range(n_folds)]
        if n_folds:
            assert image_fold is not None
            held = [image_fold == f for f in range(n_folds)]

            def held_out():
                parts = [(int(h.sum()), evaluate(d, fr["train"], steps, args.batch, args.max_nodes, h))
                         for d, h in zip(fold_models, held)]
                return dict(combined(parts), name="out-of-fold")

            def report():
                return dict(evaluate(decoder, fr["val"], steps, args.batch, args.max_nodes), name="val")

            epoch, oof, metrics = fit_depth(fold_models + [decoder], [~h for h in held] + [None], fr["train"],
                                            held_out, report, args, depth, log, f"depth {depth}")
            metrics = dict(metrics, out_of_fold=oof)
        else:
            def val_only():
                return dict(evaluate(decoder, fr["val"], steps, args.batch, args.max_nodes), name="val")

            epoch, metrics, _ = fit_depth([decoder], [None], fr["train"], val_only, lambda: None, args, depth, log,
                                          f"depth {depth}")
        state = decoder.state_dict()

        descend_all(decoder, fold_models if n_folds else None)
        bits = sum(m["bits"] for m in saved["metrics"]) + metrics["bits"]
        saved["configs"].append(decoder.config)
        saved["decoders"].append({key: v.cpu() for key, v in state.items()})
        saved["metrics"].append(dict(metrics, epoch=epoch, seconds=time.time() - t_depth))
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save(saved, args.out)
        if n_folds:
            fold_saved["configs"].append([d.config for d in fold_models])
            fold_saved["decoders"].append([{key: v.half().cpu() if v.is_floating_point() else v.cpu()
                                            for key, v in d.state_dict().items()} for d in fold_models])
            torch.save(fold_saved, folds_path)
        mse, train_mse = fr["val"].R.square().mean().item(), fr["train"].R.square().mean().item()
        log(f"depth {depth} done (epoch {epoch}, {time.time() - t_depth:.0f}s): val mse {mse:.6f} "
            f"(psnr {psnr(mse):.2f} dB) after {depth + 1} tokens, {bits:.2f} bits per image; train mse "
            f"{train_mse:.6f}{' (held-out routes)' if n_folds else ''}")


if __name__ == "__main__":
    main()
