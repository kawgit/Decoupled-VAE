"""How well do a tree's own shares predict which side real images go? Routes the val images down a train_walk.py
checkpoint's tree (its own steps), then scores each image's code under the decoder's shares, the rare side with
probability p = m / (1 + m) (what qa.py samples from): the bits each token costs, per depth and in all, beside
1 bit a token, the best constant p per depth, and the decoder's p recalibrated per depth (logit p' = a logit p + b,
fit on one half of the val images and scored on the other, both ways round). Then a calibration table: predicted p
against the share of images that went to the rare side, by bins of p and ranges of depth.

Run from the repo: python scc/share_bits.py --ckpt artifacts/<run>/model.pt
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from dvae.artifacts import artifact  # noqa: E402
from dvae.data import get_device, load_mnist  # noqa: E402
from dvae.deltatree import descend, load_tree, pixels, root_frontier, tree_steps  # noqa: E402

EPS = 1e-6


def bits(p, rare):
    """The bits [...] each token costs when the rare side has probability p."""
    return -torch.where(rare, p, 1 - p).clamp_min(EPS).log2()


def fit(z, rare, iters=50):
    """(a, b) of the logistic regression of rare [n] on z [n], by Newton's method."""
    w = torch.tensor([1.0, 0.0], dtype=torch.float64)
    Z = torch.stack([z.double(), torch.ones_like(z, dtype=torch.float64)], 1)
    y = rare.double()
    for _ in range(iters):
        q = torch.sigmoid(Z @ w)
        hess = Z.T @ (Z * (q * (1 - q))[:, None]) + 1e-6 * torch.eye(2, dtype=torch.float64)
        w = w - torch.linalg.solve(hess, Z.T @ (q - y))
    return w.float()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--limit", type=int, default=None, help="val images to score (default: all)")
    p.add_argument("--device", default=None)
    p.add_argument("--data-root", default="data")
    a = p.parse_args()
    device = get_device(a.device)
    ckpt = torch.load(a.ckpt, map_location=device, weights_only=False)
    decoders, args = load_tree(ckpt, device), ckpt["args"]
    steps, batch, tol, K = tree_steps(ckpt, len(decoders)), args["node_batch"], args.get("tol", 0.0), len(decoders)
    X = pixels(load_mnist("val", a.data_root)[:a.limit]).to(device)
    out = open(artifact("shares.txt"), "a")

    def report(text):
        print(text, flush=True)
        out.write(text + "\n")
        out.flush()

    with torch.no_grad():
        fr = root_frontier(X, ckpt["mean"].to(device))
        for decoder, depth_steps in zip(decoders, steps):
            fr = descend(decoder, fr, depth_steps, batch, tol)
    node = fr.node.cpu()
    rare, m = fr.sides.cpu()[node] > 0, fr.ms.cpu()[node].float()  # [N, K] each image's tokens
    prob = (m / (1 + m)).clamp(EPS, 1 - EPS)
    N = len(X)

    own = bits(prob, rare).mean(0)  # [K] per image
    share = rare.float().mean(0)
    constant = bits(share.clamp(EPS, 1 - EPS).expand(N, K), rare).mean(0)
    halves = torch.randperm(N, generator=torch.Generator().manual_seed(0)).chunk(2)
    z = prob.logit()
    recal = torch.zeros(K)
    for fit_on, score_on in (halves, halves[::-1]):
        for k in range(K):
            w = fit(z[fit_on, k], rare[fit_on, k])
            recal[k] += bits(torch.sigmoid(w[0] * z[score_on, k] + w[1]), rare[score_on, k]).sum() / N

    report(f"{a.ckpt}: {N} val images, {K} tokens, scored under the decoder's shares p = m / (1 + m)")
    report("bits per image: decoder {:.2f}, recalibrated {:.2f}, constant p per depth {:.2f}, uniform {}".format(
        own.sum(), recal.sum(), constant.sum(), K))
    ranges = [(lo, min(hi, K)) for lo, hi in ((1, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 256)) if lo <= K]
    report("depths    bits/token: decoder  recal  constant | rare share  mean p")
    for lo, hi in ranges:
        s = slice(lo - 1, hi)
        report(f"{lo:3d}-{hi:<3d}   {own[s].mean():19.3f}  {recal[s].mean():5.3f}  {constant[s].mean():8.3f} | "
               f"{share[s].mean():10.3f}  {prob[:, s].mean():6.3f}")
    report("per depth, bits/token decoder (recalibrated): " + " ".join(
        f"{k + 1}:{own[k]:.2f}({recal[k]:.2f})" for k in range(K)))

    edges = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
    report("calibration: for tokens with predicted p in a bin, mean p -> share that went rare (tokens)")
    for lo, hi in ranges:
        s = slice(lo - 1, hi)
        pp, rr = prob[:, s].flatten(), rare[:, s].flatten()
        cells = []
        for b0, b1 in zip(edges, edges[1:]):
            inside = (pp >= b0) & (pp < b1)
            if inside.sum() >= 20:
                cells.append(f"{pp[inside].mean():.3f}->{rr[inside].float().mean():.3f} ({int(inside.sum())})")
        report(f"  depths {lo}-{hi}: " + ", ".join(cells))
    report(f"(a code under the decoder's shares costs {own.sum() / K:.3f} bits a token; {math.log2(N):.1f} bits "
           f"would name any one of these {N} images)")


if __name__ == "__main__":
    main()
