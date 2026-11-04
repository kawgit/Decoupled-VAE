"""Compression and generation QA for a delta tree (train.py).

Compression: test images, each followed by its node's predicted mean after a spread of token counts ending with
the last (the decoder's reconstruction); the same for training images (which the tree was fit to, so the two
grids and their mse show how much it overfits); and with --train-means a third grid of the mean of the training images
in that same node (what an ideal decoder of the same tree would show; blank where the node holds none).
Generation: codes sampled token by token from the decoders' own shares (the rare side with probability
p = m / (1 + m)), shown as their predicted means after a spread of token counts ending with the last; and a
square grid of samples after every token beside the same grid of real test images, to compare at a glance.
With --prior (train_prior.py's), also square grids of codes whose sides are drawn from the prior instead, at
each --temperatures.
"""

import argparse
import os

import torch

from dvae.artifacts import artifact
from dvae.data import get_device, load_mnist
from dvae.deltatree import D, Frontier, child_key, choose, descend, load_tree, pixels, root_frontier, tree_steps
from dvae.prior import BandedPrior, SidePrior, load_prior
from dvae.viz import save_grid


@torch.no_grad()
def sample(decoders, mean, n, steps, batch, tol, generator, prior: SidePrior | BandedPrior | None = None,
           temperature=1.0):
    """n codes drawn from the decoders (each depth with its steps), and their predicted means after every token
    [n, K, D]. Each token is rare with probability m / (1 + m), or with a prior, with the prior's probability
    (its logit divided by temperature)."""
    device = mean.device
    rs, ms, rares = [], [], []  # the codes so far, as the prior reads them
    fr = root_frontier(mean.new_zeros(0, D), mean)
    fr = Frontier(R=fr.R, node=fr.node, bits=fr.bits.expand(n).clone(), deltas=fr.deltas.expand(n, 0, D),
                  sides=fr.sides.expand(n, 0), ms=fr.ms.expand(n, 0))
    out, x = [], mean.expand(n, D)
    for decoder, depth_steps in zip(decoders, steps):
        s = choose(decoder, fr, depth_steps, batch, tol)
        r, m = s.r, s.m
        p = m / (1 + m)
        if prior is not None:
            rs.append(r.half().float())  # as dump_codes.py stores them
            ms.append(m.half().float())
            unknown = torch.zeros(n, dtype=torch.bool, device=device)  # this token's side, which it doesn't read
            logits = prior(torch.stack(rs, 1), torch.stack(ms, 1), torch.stack(rares + [unknown], 1))
            p = torch.sigmoid(logits[:, -1] / temperature)
        rare = torch.rand(n, generator=generator).to(device) < p
        rares.append(rare)
        delta = torch.where(rare[:, None], r, -m[:, None] * r)
        x = x + delta
        out.append(x)
        rc = rare.cpu()
        fr = Frontier(R=fr.R, node=fr.node, bits=child_key(fr.bits, rc, fr.depth),
                      deltas=torch.cat([fr.deltas, delta.half().cpu()[:, None]], 1),
                      sides=torch.cat([fr.sides, (2 * rc.float() - 1)[:, None]], 1),
                      ms=torch.cat([fr.ms, m.cpu()[:, None]], 1),
                      seed=None if s.u is None else torch.where(rare[:, None], s.u[:, 0], s.u[:, 1]))
    return torch.stack(out, 1)


def columns(K, n):
    """n token counts from 1 to K, evenly spread and ending with K (all of them when K <= n)."""
    if K <= n:
        return list(range(1, K + 1))
    return [round(1 + (K - 1) * i / (n - 1)) for i in range(n)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="checkpoints/deltatree.pt")
    p.add_argument("-n", type=int, default=16, help="test images, and training images, to show compressed")
    p.add_argument("--score", type=int, default=1000, help="test and training images whose mse is printed")
    p.add_argument("--generate", type=int, default=16, help="codes to sample")
    p.add_argument("--prior", default=None, help="a train_prior.py checkpoint: also sample sides from it")
    p.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 0.7],
                   help="the prior's temperatures, a samples grid each")
    p.add_argument("--grid", type=int, default=8, help="side of the samples' and real images' square grids (0: none)")
    p.add_argument("--cols", type=int, default=16, help="columns per grid (compression's first is the image)")
    p.add_argument("--train-means", action="store_true",
                   help="also show each test image's node as its training images' mean (sends all of them down)")
    p.add_argument("--out", default=None, help="folder for the pictures (default: the run's artifacts folder)")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="data")
    args = p.parse_args()
    device = get_device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    decoders = load_tree(ckpt, device)
    a, mean = ckpt["args"], ckpt["mean"].to(device)
    steps, batch, K, tol = tree_steps(ckpt, len(decoders)), a["node_batch"], len(decoders), a.get("tol", 0.0)
    args.out = args.out or artifact("")
    os.makedirs(args.out, exist_ok=True)

    # the test and training images' codes and predicted means after each token (and with --train-means every
    # training image's code)
    count = max(args.n, args.score)
    X = pixels(load_mnist("test", args.data_root)[:count]).to(device)
    Xt = pixels(load_mnist("train", args.data_root)[:count]).to(device)
    T = pixels(load_mnist("train", args.data_root)).to(device) if args.train_means else X[:0]
    fr = {"test": root_frontier(X, mean), "seen": root_frontier(Xt, mean)}
    if args.train_means:
        fr["train"] = root_frontier(T, mean)
    preds, preds_seen, empirical = [X], [Xt], [X[:args.n]]
    for decoder, depth_steps in zip(decoders, steps):
        fr = {split: descend(decoder, f, depth_steps, batch, tol) for split, f in fr.items()}
        preds.append(X - fr["test"].R)
        preds_seen.append(Xt - fr["seen"].R)
        if args.train_means:
            code = fr["test"].bits[fr["test"].node.cpu()][:args.n]
            train_code = fr["train"].bits[fr["train"].node.cpu()]
            row = []
            for c in code.tolist():
                inside = (train_code == c).nonzero()[:, 0].to(device)
                row.append(T[inside].mean(0) if len(inside) else torch.zeros(D, device=device))
            empirical.append(torch.stack(row))
    for name, ps, images in (("test", preds, X), ("training", preds_seen, Xt)):
        mse = [(ps[k][:args.score] - images[:args.score]).square().mean().item() for k in range(1, K + 1)]
        print(f"{name} mse over {min(args.score, len(images))} images after 1..{K} tokens: "
              + ", ".join(f"{v:.4f}" for v in mse))
    preds, preds_seen = [x[:args.n] for x in preds], [x[:args.n] for x in preds_seen]

    shown = columns(K, args.cols - 1)
    cols = [0] + shown
    path = os.path.join(args.out, "compress.png")
    save_grid(torch.stack(preds, 1)[:, cols].clamp(0, 1).reshape(-1, 1, 28, 28), path, nrow=len(cols))
    print(f"{path}: rows are test images, then the decoder's predicted mean after {shown} tokens")
    path = os.path.join(args.out, "compress_train.png")
    save_grid(torch.stack(preds_seen, 1)[:, cols].clamp(0, 1).reshape(-1, 1, 28, 28), path, nrow=len(cols))
    print(f"{path}: rows are training images, then the decoder's predicted mean after {shown} tokens")
    if args.train_means:
        counts = [int((fr["train"].bits[fr["train"].node.cpu()] == c).sum()) for c in
                  fr["test"].bits[fr["test"].node.cpu()][:args.n].tolist()]
        print(f"training images in each test image's node after {K} tokens: {counts}")
        path = os.path.join(args.out, "compress_train_means.png")
        save_grid(torch.stack(empirical, 1)[:, cols].clamp(0, 1).reshape(-1, 1, 28, 28), path, nrow=len(cols))
        print(f"{path}: rows are test images, then the mean of the training images in the same node after "
              f"{shown} tokens")

    gen = torch.Generator().manual_seed(args.seed)
    xs = sample(decoders, mean, max(args.generate, args.grid ** 2), steps, batch, tol, gen)
    shown = columns(K, args.cols)
    path = os.path.join(args.out, "generate.png")
    save_grid(xs[:args.generate, [k - 1 for k in shown]].clamp(0, 1).reshape(-1, 1, 28, 28), path, nrow=len(shown))
    print(f"{path}: rows are sampled codes, shown as their predicted mean after {shown} tokens")
    if args.grid:
        path = os.path.join(args.out, "samples.png")
        save_grid(xs[:args.grid ** 2, -1].clamp(0, 1).reshape(-1, 1, 28, 28), path, nrow=args.grid)
        print(f"{path}: {args.grid ** 2} sampled codes, shown as their predicted mean after all {K} tokens")
        real = pixels(load_mnist("test", args.data_root)[:args.grid ** 2])
        path = os.path.join(args.out, "real.png")
        save_grid(real.reshape(-1, 1, 28, 28), path, nrow=args.grid)
        print(f"{path}: the first {args.grid ** 2} test images, for comparison")
    if args.prior:
        prior = load_prior(torch.load(args.prior, map_location=device, weights_only=False), device)
        assert prior.config["depths"] >= K, f"the prior knows {prior.config['depths']} tokens, the tree has {K}"
        for t in args.temperatures:
            gen = torch.Generator().manual_seed(args.seed)
            xs = sample(decoders, mean, max(args.grid, 1) ** 2, steps, batch, tol, gen, prior, t)
            path = os.path.join(args.out, f"samples_prior_t{t:g}.png")
            save_grid(xs[:, -1].clamp(0, 1).reshape(-1, 1, 28, 28), path, nrow=max(args.grid, 1))
            print(f"{path}: {len(xs)} codes sampled from the prior at temperature {t:g}, after all {K} tokens")


if __name__ == "__main__":
    main()
