"""Test bits per dimension of a delta tree (train.py), after every code length up to the checkpoint's.

A code of k tokens costs its tokens' bits under the decoders' own shares (the rare side's p = m / (1 + m)).
Its pixels are coded given the code's predicted mean by a categorical over the 256 grey levels that depends
only on that mean, shared by every pixel: the mean is cut into --buckets bins and each bin has its own
distribution over levels, counted on the held-out (val) images' leaves at that k with --alpha added to every
count. The test images, unseen by everything, are scored as -log2 p(code) - log2 p(x | code), the one-code
upper bound on -log2 p(x), in bits per pixel.
"""

import argparse
import json
import math
import os

import torch

from dvae.data import load_mnist
from dvae.deltatree import D, load_tree, pixels, root_frontier, split


def bucket(mu, buckets):
    return (mu.clamp(0, 1) * buckets).long().clamp(max=buckets - 1)


def fit_levels(mu, levels, buckets, alpha):
    """log2 p(level | the predicted mean's bucket) [buckets, 256], counted from means mu and levels [N, D]."""
    counts = torch.bincount((bucket(mu, buckets) * 256 + levels).flatten(), minlength=buckets * 256)
    counts = counts.view(buckets, 256).float() + alpha
    return (counts / counts.sum(1, keepdim=True)).log2()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="checkpoints/deltatree.pt")
    p.add_argument("--buckets", type=int, default=64, help="bins of the predicted mean the level model conditions on")
    p.add_argument("--alpha", type=float, default=0.5, help="added to every level count")
    p.add_argument("--limit-test", type=int, default=None)
    p.add_argument("--results", default="results/deltatree_benchmark")
    p.add_argument("--device", default=None)
    p.add_argument("--data-root", default="data")
    args = p.parse_args()
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    a, mean = ckpt["args"], ckpt["mean"].to(device)
    steps, batch = a["lloyd_steps"], a["node_batch"]
    images = {"val": load_mnist("val", args.data_root)[:a["limit_val"]],
              "test": load_mnist("test", args.data_root)[:args.limit_test]}
    X = {s: pixels(im).to(device) for s, im in images.items()}
    levels = {s: im.long().flatten(1).to(device) for s, im in images.items()}
    fr = {s: root_frontier(x, mean) for s, x in X.items()}
    code = {s: torch.zeros(len(x), device=device) for s, x in X.items()}

    rows = []
    decoders = load_tree(ckpt, device)
    for k in range(len(decoders) + 1):
        if k:
            decoder = decoders[k - 1]
            for s in fr:
                fr[s], token_bits, _ = split(decoder, fr[s], steps, batch, a.get("tol", 0.0))
                code[s] += token_bits
        mu = {s: X[s] - fr[s].R for s in fr}
        table = fit_levels(mu["val"], levels["val"], args.buckets, args.alpha)
        leaf = -table[bucket(mu["test"], args.buckets), levels["test"]].sum(1)
        total = code["test"] + leaf
        rows.append(dict(tokens=k, code_bits=code["test"].mean().item(), leaf_bpd=leaf.mean().item() / D,
                         bpd=total.mean().item() / D, nats=total.mean().item() * math.log(2),
                         mse=fr["test"].R.square().mean().item()))
        r = rows[-1]
        print(f"{k:2d} tokens: {r['bpd']:.4f} bits/dim ({r['nats']:.1f} nats/image): code {r['code_bits']:.2f} bits, "
              f"pixels {r['leaf_bpd']:.4f} bits/dim, test mse {r['mse']:.5f}", flush=True)

    best = min(rows, key=lambda r: r["bpd"])
    print(f"best: {best['bpd']:.4f} bits/dim at {best['tokens']} tokens over {len(X['test'])} test images")
    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
    with open(args.results + ".json", "w") as f:
        json.dump(dict(ckpt=args.ckpt, buckets=args.buckets, alpha=args.alpha, test_images=len(X["test"]),
                       bound="one code: -log2 p(z(x)) - log2 p(x | z(x))", rows=rows), f, indent=2)
    with open(args.results + ".md", "w") as f:
        f.write("| tokens | test bpd | nats/image | code bits | pixel bpd | test mse |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['tokens']} | {r['bpd']:.4f} | {r['nats']:.1f} | {r['code_bits']:.2f} | "
                    f"{r['leaf_bpd']:.4f} | {r['mse']:.5f} |\n")
    print(f"wrote {args.results}.json and .md")


if __name__ == "__main__":
    main()
