"""Compression and generation QA for a delta tree (train.py).

Compression: test images, each followed by its node's predicted mean after every token (the decoder's
reconstruction), and in a second grid by the mean of the training images in that same node (what an ideal
decoder of the same tree would show; blank where the node holds none). Generation: codes sampled token by
token from the decoders' own shares (the rare side with probability p = m / (1 + m)), shown as their
predicted means after every token.
"""

import argparse
import os

import torch

from dvae.artifacts import artifact
from dvae.data import get_device, load_mnist
from dvae.deltatree import D, Frontier, child_key, choose, descend, load_tree, pixels, root_frontier
from dvae.viz import save_grid


@torch.no_grad()
def sample(decoders, mean, n, steps, batch, tol, generator):
    """n codes drawn from the decoders, and their predicted means after every token [n, K, D]."""
    device = mean.device
    fr = root_frontier(mean.new_zeros(0, D), mean)
    fr = Frontier(R=fr.R, node=fr.node, bits=fr.bits.expand(n).clone(), deltas=fr.deltas.expand(n, 0, D),
                  sides=fr.sides.expand(n, 0), ms=fr.ms.expand(n, 0))
    out, x = [], mean.expand(n, D)
    for decoder in decoders:
        s = choose(decoder, fr, steps, batch, tol)
        r, m = s.r, s.m
        rare = torch.rand(n, generator=generator).to(device) < m / (1 + m)
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="checkpoints/deltatree.pt")
    p.add_argument("-n", type=int, default=16, help="test images to compress")
    p.add_argument("--generate", type=int, default=16, help="codes to sample")
    p.add_argument("--out", default=None, help="folder for the pictures (default: the run's artifacts folder)")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="data")
    args = p.parse_args()
    device = get_device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    decoders = load_tree(ckpt, device)
    a, mean = ckpt["args"], ckpt["mean"].to(device)
    steps, batch, K, tol = a["lloyd_steps"], a["node_batch"], len(decoders), a.get("tol", 0.0)
    args.out = args.out or artifact("")
    os.makedirs(args.out, exist_ok=True)

    # every test image's and every training image's code and predicted mean after each token
    X = pixels(load_mnist("test", args.data_root)[:args.n]).to(device)
    T = pixels(load_mnist("train", args.data_root)).to(device)
    fr = {"test": root_frontier(X, mean), "train": root_frontier(T, mean)}
    preds, empirical = [X], [X]
    for decoder in decoders:
        fr = {split: descend(decoder, f, steps, batch, tol) for split, f in fr.items()}
        preds.append(X - fr["test"].R)
        code = fr["test"].bits[fr["test"].node.cpu()]
        train_code = fr["train"].bits[fr["train"].node.cpu()]
        row = []
        for c in code.tolist():
            inside = (train_code == c).nonzero()[:, 0].to(device)
            row.append(T[inside].mean(0) if len(inside) else torch.zeros(D, device=device))
        empirical.append(torch.stack(row))
    counts = [int((fr["train"].bits[fr["train"].node.cpu()] == c).sum()) for c in
              fr["test"].bits[fr["test"].node.cpu()].tolist()]
    mse = [(preds[k] - X).square().mean().item() for k in range(1, K + 1)]
    print(f"test mse after 1..{K} tokens: " + ", ".join(f"{v:.4f}" for v in mse))
    print(f"training images in each test image's node after {K} tokens: {counts}")

    path = os.path.join(args.out, "compress.png")
    save_grid(torch.stack(preds, 1).clamp(0, 1).reshape(-1, 1, 28, 28), path, nrow=1 + K)
    print(f"{path}: rows are test images, then the decoder's predicted mean after 1..{K} tokens")
    path = os.path.join(args.out, "compress_train_means.png")
    save_grid(torch.stack(empirical, 1).clamp(0, 1).reshape(-1, 1, 28, 28), path, nrow=1 + K)
    print(f"{path}: rows are test images, then the mean of the training images in the same node after 1..{K} tokens")

    gen = torch.Generator().manual_seed(args.seed)
    xs = sample(decoders, mean, args.generate, steps, batch, tol, gen)
    path = os.path.join(args.out, "generate.png")
    save_grid(xs.clamp(0, 1).reshape(-1, 1, 28, 28), path, nrow=K)
    print(f"{path}: rows are sampled codes, shown as their predicted mean after 1..{K} tokens")


if __name__ == "__main__":
    main()
