"""Compression QA for the autoencoder (train_autoencoder.py), on training and test images alike, so the two show
how well it generalizes.

compress_train.png and compress_test.png: images, each followed by its node's predicted mean (its reconstruction)
after a spread of token counts ending with the last. mse.png: the mse per pixel after every token count, over
--score training and --score test images. With --train-means, compress_train_means.png: each test image's node
shown as the mean of the training images in it (what an ideal decoder of the same tree would show; blank where
the node holds none), which sends every training image down the tree.

Run from the repo: python qa_autoencoder.py --ckpt artifacts/<run>/model.pt   (pictures go to artifacts/<job>/)
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from dvae.artifacts import artifact  # noqa: E402
from dvae.data import get_device, load_mnist  # noqa: E402
from dvae.deltatree import D, descend, pixels, root_frontier  # noqa: E402
from dvae.sampling import columns, load_autoencoder  # noqa: E402
from dvae.viz import save_grid  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="train_autoencoder.py's checkpoint")
    p.add_argument("-n", type=int, default=16, help="training and test images shown compressed")
    p.add_argument("--score", type=int, default=1000, help="training and test images whose mse is measured")
    p.add_argument("--cols", type=int, default=16, help="columns per grid, the first being the image")
    p.add_argument("--train-means", action="store_true",
                   help="also show each test image's node as its training images' mean (sends all of them down)")
    p.add_argument("--out", default=None, help="folder for the pictures (default: the run's artifacts folder)")
    p.add_argument("--device", default=None)
    p.add_argument("--data-root", default="data")
    args = p.parse_args()
    device = get_device(args.device)
    decoders, mean, steps, batch, tol = load_autoencoder(args.ckpt, device)
    K = len(decoders)
    out = args.out or artifact("")
    os.makedirs(out, exist_ok=True)

    count = max(args.n, args.score)
    X = {s: pixels(load_mnist(s, args.data_root)[:count]).to(device) for s in ("train", "test")}
    T = pixels(load_mnist("train", args.data_root)).to(device) if args.train_means else None
    fr = {s: root_frontier(x, mean) for s, x in X.items()}
    if T is not None:
        fr["all"] = root_frontier(T, mean)
    preds = {s: [x] for s, x in X.items()}  # each split's images, then their predicted means after every token
    empirical = [X["test"][:args.n]]
    for decoder, depth_steps in zip(decoders, steps):
        fr = {s: descend(decoder, f, depth_steps, batch, tol) for s, f in fr.items()}
        for s, x in X.items():
            preds[s].append(x - fr[s].R)
        if T is not None:
            code = fr["test"].bits[fr["test"].node.cpu()][:args.n]
            train_code = fr["all"].bits[fr["all"].node.cpu()]
            row = []
            for c in code.tolist():
                inside = (train_code == c).nonzero()[:, 0].to(device)
                row.append(T[inside].mean(0) if len(inside) else torch.zeros(D, device=device))
            empirical.append(torch.stack(row))

    mse = {s: [(preds[s][k][:args.score] - x[:args.score]).square().mean().item() for k in range(1, K + 1)]
           for s, x in X.items()}
    for s, values in mse.items():
        print(f"{s} mse over {min(args.score, len(X[s]))} images after 1..{K} tokens: "
              + ", ".join(f"{v:.4f}" for v in values))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, color in (("test", "#2a78d6"), ("train", "#eb6834")):
        ax.plot(range(1, K + 1), mse[s], color=color, linewidth=1.6, label=f"{s} ({mse[s][-1]:.4f} after {K})")
    ax.set_yscale("log")
    ax.set_xlabel("tokens")
    ax.set_ylabel("mse per pixel")
    ax.set_title("Reconstruction error by code length", loc="left")
    ax.grid(True, color="#e2e2df")
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    path = os.path.join(out, "mse.png")
    fig.savefig(path, dpi=150, facecolor="white")
    print(f"{path}: mse per pixel by code length, training and test images")

    shown = columns(K, args.cols - 1)
    cols = [0] + shown
    for s, name in (("train", "training"), ("test", "test")):
        path = os.path.join(out, f"compress_{s}.png")
        grid = torch.stack([x[:args.n] for x in preds[s]], 1)[:, cols]
        save_grid(grid.clamp(0, 1).reshape(-1, 1, 28, 28), path, nrow=len(cols))
        print(f"{path}: rows are {name} images, then their predicted mean after {shown} tokens")
    if T is not None:
        path = os.path.join(out, "compress_train_means.png")
        save_grid(torch.stack(empirical, 1)[:, cols].clamp(0, 1).reshape(-1, 1, 28, 28), path, nrow=len(cols))
        print(f"{path}: rows are test images, then the mean of the training images in the same node after "
              f"{shown} tokens")


if __name__ == "__main__":
    main()
