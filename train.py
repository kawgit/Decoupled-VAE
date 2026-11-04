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

--resume deepens a trained tree: it replays the checkpoint's decoders to rebuild both frontiers, then trains
the depths after them up to --max-depth. The settings that define the tree (the data, the root mean and the
Lloyd steps) come from the checkpoint; the training settings are the ones given now. Each depth
draws its batches from its own seed, so a resumed run trains the same as one that never stopped.
"""

import argparse
import math
import os
import time
from typing import Any

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")  # ~6.5 GB: fail instead of swapping
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")

import torch  # noqa: E402

from dvae.artifacts import artifact  # noqa: E402
from dvae.data import get_device, load_mnist  # noqa: E402
from dvae.logs import RunLog  # noqa: E402
from dvae.deltatree import (D, DeltaDecoder, Frontier, by_nodes, chunk_sse, descend, node_order,  # noqa: E402
                            pixels, root_frontier, top_direction)


TREE_ARGS = ("lloyd_steps", "limit_train", "limit_val", "data_root")


def train_chunks(fr, size, gen):
    """An epoch's batches: images grouped by node, nodes in random order, images in random order within a node,
    cut every size images (a node may straddle two batches; its gradient still sums exactly)."""
    perm = torch.randperm(len(fr.R), generator=gen)
    rank = torch.randperm(len(fr.bits), generator=gen)[fr.node.cpu()[perm]]
    perm = perm[torch.argsort(rank, stable=True)].to(fr.R.device)
    return [perm[i:i + size] for i in range(0, len(perm), size)]


@torch.no_grad()
def evaluate(decoder, fr, steps, size, cap):
    """The objective per image and pixel (averaged over steps), and the last step's split: mse before and after
    (per pixel) and the token's bits per image."""
    decoder.eval()
    order, _ = node_order(fr)
    total = dict(sse=0.0, before=0.0, after=0.0, bits=0.0)
    for i in range(0, len(order), size):
        for imgs in by_nodes(order[i:i + size], fr.node[order[i:i + size]], cap):
            sse, _, stats, _, _ = chunk_sse(decoder, fr, imgs, steps, select=True)
            assert stats is not None
            total["sse"] += sse.item()
            for key, value in stats.items():
                total[key] += value.item()
    N = len(order)
    return dict(loss=total["sse"] / (N * D), mse_before=total["before"] / (N * D),
                mse_after=total["after"] / (N * D), bits=total["bits"] / N)


def psnr(mse):
    return -10 * math.log10(mse)


def train_epoch(decoder, opt, sched, fr, gen, args, log):
    """One epoch of decoder on frontier fr. Returns the objective's sum over its images."""
    steps, norm = args.lloyd_steps, D  # each node's objective is already averaged over its Lloyd steps
    decoder.train()
    total, last = fr.R.new_zeros(()), time.time()
    batches = train_chunks(fr, args.batch, gen)
    for step, imgs in enumerate(batches):
        opt.zero_grad(set_to_none=True)
        loss = fr.R.new_zeros(())
        for piece in by_nodes(imgs, fr.node[imgs], args.max_nodes):
            sse, _, _, _, _ = chunk_sse(decoder, fr, piece, steps)
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


def fit(decoder, fr, val_fr, args, depth, log):
    """Train decoder on frontier fr, keeping the epoch with the best mse after the split on val_fr and early
    stopping on it. Returns the kept epoch and its val metrics."""
    steps, norm = args.lloyd_steps, D
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup_steps))
    gen = torch.Generator().manual_seed(args.seed * 1000 + depth)
    best: tuple[float, int, dict | None, dict | None] = (math.inf, -1, None, None)
    for epoch in range(args.epochs):
        t0 = time.time()
        total = train_epoch(decoder, opt, sched, fr, gen, args, log)
        t1 = time.time()
        val = evaluate(decoder, val_fr, steps, args.batch, args.max_nodes)
        improved = val["mse_after"] < best[0]
        log(f"  depth {depth} epoch {epoch:2d}: val mse {val['mse_before']:.6f} -> {val['mse_after']:.6f} "
            f"(psnr {psnr(val['mse_after']):.2f} dB), {val['bits']:.3f} bits; objective train "
            f"{total / (len(fr.R) * norm):.6f}, val {val['loss']:.6f} "
            f"({t1 - t0:.0f}s train, {time.time() - t1:.0f}s val){'  *' if improved else ''}")
        if improved:
            best = (val["mse_after"], epoch, {key: v.detach().clone() for key, v in decoder.state_dict().items()}, val)
        elif epoch - best[1] >= args.patience:
            break
    assert best[2] is not None and best[3] is not None
    decoder.load_state_dict(best[2])
    decoder.eval()
    return best[1], best[3]


def load_decoder(cfg, state, device):
    decoder = DeltaDecoder(**cfg).to(device)
    decoder.load_state_dict(state)
    return decoder.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-depth", type=int, default=32, help="tokens per code")
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
    p.add_argument("--out", default=None, help="checkpoint (default: model.pt in the run's artifacts folder)")
    p.add_argument("--resume", nargs="?", const="", default=None, metavar="CKPT",
                   help="deepen CKPT (default: --out): replay its decoders, then train the depths after them")
    p.add_argument("--log", default=None, help="summary log (default: run.log in the run's artifacts folder)")
    p.add_argument("--log-interval", type=float, default=10.0, help="seconds between step logs")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="data")
    args = p.parse_args()
    args.out, args.log = args.out or artifact("model.pt"), args.log or artifact("run.log")

    device = get_device(args.device)
    log = RunLog(args.log, device)

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
    fr: dict[str, Frontier] = {split: root_frontier(x, mean) for split, x in X.items()}
    steps = args.lloyd_steps

    def descend_all(decoder):
        for split in fr:
            fr[split] = descend(decoder, fr[split], steps, args.node_batch)

    saved: dict[str, Any] = dict(args=vars(args), mean=mean.cpu(), configs=[], decoders=[], metrics=[])
    if ckpt is not None:
        source = args.resume or args.out
        for depth, (cfg, state) in enumerate(zip(ckpt["configs"], ckpt["decoders"])):
            descend_all(load_decoder(cfg, state, device))
            mse = fr["val"].R.square().mean().item()
            log(f"replayed depth {depth}: val mse {mse:.6f} (psnr {psnr(mse):.2f} dB) after {depth + 1} tokens, "
                f"train mse {fr['train'].R.square().mean().item():.6f}")
        saved.update(configs=ckpt["configs"], decoders=ckpt["decoders"], metrics=ckpt["metrics"])
        log(f"resumed {len(ckpt['decoders'])} depths from {source}, saving to {args.out}")

    for depth in range(len(saved["decoders"]), args.max_depth):
        t_depth = time.time()
        scale = fr["train"].R.square().mean().sqrt().item()
        log(f"depth {depth}: {len(fr['train'].bits)} train nodes, {len(fr['val'].bits)} val nodes, "
            f"remainder rms {scale:.4f}")
        torch.manual_seed(args.seed + depth)
        decoder = DeltaDecoder(depth, scale, args.width, args.layers, args.heads).to(device)
        with torch.no_grad():
            decoder.start.copy_(top_direction(fr["train"].R))
        epoch, metrics = fit(decoder, fr["train"], fr["val"], args, depth, log)
        state = decoder.state_dict()

        descend_all(decoder)
        bits = sum(m["bits"] for m in saved["metrics"]) + metrics["bits"]
        saved["configs"].append(decoder.config)
        saved["decoders"].append({key: v.cpu() for key, v in state.items()})
        saved["metrics"].append(dict(metrics, epoch=epoch, seconds=time.time() - t_depth))
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save(saved, args.out)
        mse, train_mse = fr["val"].R.square().mean().item(), fr["train"].R.square().mean().item()
        log(f"depth {depth} done (epoch {epoch}, {time.time() - t_depth:.0f}s): val mse {mse:.6f} "
            f"(psnr {psnr(mse):.2f} dB) after {depth + 1} tokens, {bits:.2f} bits per image; train mse "
            f"{train_mse:.6f}")


if __name__ == "__main__":
    main()
