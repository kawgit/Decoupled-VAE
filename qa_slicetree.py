"""Compression and generation QA for a slice tree (see README).

Analytic compression (needs no decoder): test images, each followed by the mean of the training images in
its node after k tokens, for each k in --depths, and for each k a grid of each test image followed by -m
training images drawn from its node (the ideal decoder's posterior; blank where the node holds none).
With a trained Gaussian decoder, also: each test image followed by the decoder's node means, and codes sampled
from the prior, shown as their leaves' decoder means. With a categorical one (train_slicetree_levels.py): each
test image with its leaf's mean and a sample, and sampled codes' leaf means and samples.
"""

import argparse
import os

import torch

from dvae.data import load_mnist
from dvae.slicetree import D, Root, SliceModel, decode, walk_model
from dvae.viz import save_grid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="checkpoints/slicetree.pt")
    p.add_argument("-n", type=int, default=16, help="test images to compress")
    p.add_argument("-m", type=int, default=8, help="training images drawn from each node")
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 16, 24, 32])
    p.add_argument("--generate", type=int, default=64)
    p.add_argument("--out", default="outputs/slicetree")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="data")
    args = p.parse_args()
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    models = {}
    for name in ("encoder", "prior", "decoder", "levels"):
        if ckpt.get(name) is not None:
            models[name] = SliceModel(**ckpt["configs"][name]).to(device)
            models[name].load_state_dict(ckpt[name])
            models[name].eval()
    root = Root(**ckpt["root"])
    a = ckpt["args"]
    K = a["K"]
    walk = dict(K=K, starts=a["starts"], start_depths=a["start_depths"], iters=a["iters"])
    depths = [k for k in args.depths if k <= K]
    os.makedirs(args.out, exist_ok=True)

    images = load_mnist("test", args.data_root)[:args.n]
    X = ((images.float() + 0.5) / 256).flatten(1).to(device) - root.center
    train = ((load_mnist("train", args.data_root).float() + 0.5) / 256).flatten(1).to(device) - root.center
    tokens, _, H, o = walk_model(models["encoder"], models["prior"], X=X, **walk)

    # the training images in each test image's node, at every depth: [N, K] per image
    gen = torch.Generator().manual_seed(args.seed)
    means, draws = [], {k: [] for k in depths}
    for i in range(len(X)):
        inside = ((train @ H[i].T) >= o[i]).cumprod(1).bool()
        row = []
        for k in depths:
            idx = inside[:, k - 1].nonzero()[:, 0]
            if len(idx):
                row.append(train[idx].mean(0))
                pick = idx[torch.randint(len(idx), (args.m,), generator=gen).to(device)]
                draws[k].append(torch.cat([X[i:i + 1], train[pick]]))
            else:
                row.append(torch.full((D,), -1.0, device=device))
                draws[k].append(torch.cat([X[i:i + 1], torch.full((args.m, D), -1.0, device=device)]))
        means.append(torch.stack([X[i]] + row))
        print(f"image {i:2d}: training images in its node at depths {depths}: "
              + ", ".join(str(int(inside[:, k - 1].sum())) for k in depths))
    path = os.path.join(args.out, "analytic.png")
    save_grid((torch.stack(means) + root.center).clamp(0, 1).reshape(-1, 1, 28, 28), path, nrow=1 + len(depths))
    print(f"{path}: rows are originals, then the mean of the training images in their node after {depths} tokens")
    for k in depths:
        path = os.path.join(args.out, f"analytic_k{k:02d}.png")
        save_grid((torch.cat(draws[k]) + root.center).clamp(0, 1).reshape(-1, 1, 28, 28), path, nrow=1 + args.m)
    print(f"{args.out}/analytic_kNN.png: each original, then {args.m} training images drawn from its node after k tokens")

    gen_tokens, _, gen_H, gen_o = walk_model(models["encoder"], models["prior"], n=args.generate,
                                             generator=torch.Generator().manual_seed(args.seed), **walk)
    if "levels" in models:  # the categorical decoder, at the leaves
        def leaves(H, o):
            f = models["levels"].features(models["levels"].memory(H[:, :-1], o[:, :-1]), H[:, -1:], o[:, -1:])
            p = models["levels"].levels(f)[:, 0].softmax(-1)                        # [n, D, 256]
            mean = p @ (torch.arange(256, device=device, dtype=torch.float) / 255)
            sample = torch.multinomial(p.flatten(0, 1), 1, generator=gen_d).view(len(H), D).float() / 255
            return mean, sample
        gen_d = torch.Generator(device=device).manual_seed(args.seed)
        with torch.no_grad():
            mean, sample = leaves(H, o)
            path = os.path.join(args.out, "compress_levels.png")
            save_grid(torch.stack([X + root.center, mean, sample], 1).clamp(0, 1).reshape(-1, 1, 28, 28), path, nrow=3)
            print(f"{path}: rows are originals, then the categorical decoder's leaf mean and a sample from it")
            mean, sample = leaves(gen_H, gen_o)
            path = os.path.join(args.out, "generate_levels.png")
            save_grid(mean.clamp(0, 1).reshape(-1, 1, 28, 28), path, nrow=8)
            save_grid(sample.reshape(-1, 1, 28, 28), path.replace(".png", "_samples.png"), nrow=8)
            print(f"{path} (and _samples): the categorical decoder's leaf means (and samples) of {args.generate} "
                  "codes sampled from the prior")

    if "decoder" not in models:
        print("no trained Gaussian decoder in the checkpoint: its compression and generation skipped")
        return
    node_means, _, _ = decode(models["decoder"], root, tokens, H, o)
    rows = torch.stack([X] + [node_means[:, k] for k in depths], 1) + root.center
    path = os.path.join(args.out, "compress.png")
    save_grid(rows.clamp(0, 1).reshape(-1, 1, 28, 28), path, nrow=1 + len(depths))
    print(f"{path}: rows are originals, then the decoder's node means after {depths} tokens")

    node_means, _, _ = decode(models["decoder"], root, gen_tokens, gen_H, gen_o)
    path = os.path.join(args.out, "generate.png")
    save_grid((node_means[:, -1] + root.center).clamp(0, 1).reshape(-1, 1, 28, 28), path, nrow=8)
    print(f"{path}: the decoder's leaf means of {args.generate} codes sampled from the prior")


if __name__ == "__main__":
    main()
