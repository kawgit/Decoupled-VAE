"""Generation QA for the prior (train_prior.py): codes sampled token by token down the autoencoder's tree, each
side drawn from the prior, shown as their nodes' predicted mean images.

generate.png: --generate codes, each shown after a spread of token counts ending with the last. samples_t<T>.png:
a --grid x --grid square of codes after every token, one per temperature (the prior's logits divided by T).
samples_own.png: the same square with sides drawn from the autoencoder's own shares (m / (1 + m)) instead, as
the baseline the prior improves on; every square uses the same random draws, so a cell is the same sample under
different side probabilities. real.png: the same square of real test images.

Run from the repo: python qa_prior.py --prior artifacts/<prior job>/prior.pt [--ckpt artifacts/<run>/model.pt]
"""

import argparse
import os

import torch

from dvae.artifacts import artifact
from dvae.data import get_device, load_mnist
from dvae.deltatree import pixels
from dvae.prior import load_prior
from dvae.sampling import columns, load_autoencoder, sample
from dvae.viz import save_grid


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prior", required=True, help="train_prior.py's checkpoint")
    p.add_argument("--ckpt", default=None, help="the autoencoder (default: the one the prior was trained on)")
    p.add_argument("--generate", type=int, default=16, help="codes shown token by token")
    p.add_argument("--grid", type=int, default=8, help="side of the square grids")
    p.add_argument("--cols", type=int, default=16, help="columns of generate.png")
    p.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 0.85, 0.7])
    p.add_argument("--out", default=None, help="folder for the pictures (default: the run's artifacts folder)")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="data")
    args = p.parse_args()
    device = get_device(args.device)
    pckpt = torch.load(args.prior, map_location=device, weights_only=False)
    ckpt = args.ckpt or pckpt.get("autoencoder")
    assert ckpt, "this prior's checkpoint doesn't name its autoencoder: give --ckpt"
    decoders, mean, steps, batch, tol = load_autoencoder(ckpt, device)
    prior = load_prior(pckpt, device)
    K = len(decoders)
    assert prior.config["depths"] >= K, f"the prior knows {prior.config['depths']} tokens, the autoencoder has {K}"
    out = args.out or artifact("")
    os.makedirs(out, exist_ok=True)
    n = max(args.generate, args.grid ** 2)

    def draw(prior_or_none, t=1.0):
        return sample(decoders, mean, n, steps, batch, tol, torch.Generator().manual_seed(args.seed), prior_or_none,
                      t)[0]

    for t in args.temperatures:
        xs = draw(prior, t)
        path = os.path.join(out, f"samples_t{t:g}.png")
        save_grid(xs[:args.grid ** 2, -1].clamp(0, 1).reshape(-1, 1, 28, 28), path, nrow=args.grid)
        print(f"{path}: {args.grid ** 2} codes sampled from the prior at temperature {t:g}, after all {K} tokens")
        if t == args.temperatures[0]:
            shown = columns(K, args.cols)
            path = os.path.join(out, "generate.png")
            grid = xs[:args.generate, [k - 1 for k in shown]]
            save_grid(grid.clamp(0, 1).reshape(-1, 1, 28, 28), path, nrow=len(shown))
            print(f"{path}: rows are codes sampled from the prior at temperature {t:g}, shown as their predicted "
                  f"mean after {shown} tokens")
    xs = draw(None)
    path = os.path.join(out, "samples_own.png")
    save_grid(xs[:args.grid ** 2, -1].clamp(0, 1).reshape(-1, 1, 28, 28), path, nrow=args.grid)
    print(f"{path}: the same draws with the autoencoder's own shares, after all {K} tokens")
    path = os.path.join(out, "real.png")
    save_grid(pixels(load_mnist("test", args.data_root)[:args.grid ** 2]).reshape(-1, 1, 28, 28), path,
              nrow=args.grid)
    print(f"{path}: the first {args.grid ** 2} test images, for comparison")


if __name__ == "__main__":
    main()
