"""Draw training images' paths as train_encoder.py's prefix cache (--cache-depths, dvae/persist.py) holds them:
each image, then its reconstruction (the data mean plus its cached deltas) after a spread of depths, blank past
the depths cached for it; and with --deltas a second grid of the deltas themselves (grey zero, each image's
scaled to its largest). The cache's files are .npy, so np.load(path, mmap_mode="r") also opens any of them.

Run from the repo: python view_cache.py --cache artifacts/<run>/cache   (the run's model.pt beside the cache
names the dataset and the mean; pictures go to the job's artifacts folder, or --out)
"""

import argparse
import json
import os

import numpy as np
import torch

from dvae.artifacts import artifact
from dvae.data import SHAPES, load_images
from dvae.deltatree import pixels
from dvae.sampling import columns
from dvae.viz import save_grid


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True, help="the cache's folder")
    p.add_argument("--ckpt", default=None, help="the run's model.pt (default: the one beside the cache)")
    p.add_argument("--images", type=int, nargs="*", default=None, help="training image indices (default: -n cached)")
    p.add_argument("-n", type=int, default=16, help="cached images drawn at random when --images is not given")
    p.add_argument("--cols", type=int, default=16, help="columns per grid, the first being the image")
    p.add_argument("--deltas", action="store_true", help="also draw each depth's delta")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="folder for the pictures (default: the job's artifacts folder)")
    p.add_argument("--data-root", default="data")
    args = p.parse_args()

    ckpt_path = args.ckpt or os.path.join(os.path.dirname(os.path.abspath(args.cache)), "model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    dataset, mean = ckpt["args"].get("dataset", "mnist"), ckpt["mean"].float()
    shape = SHAPES[dataset]
    with open(os.path.join(args.cache, "meta.json")) as f:
        meta = json.load(f)

    def load(name):
        return np.load(os.path.join(args.cache, f"{name}.npy"), mmap_mode="r")

    valid = torch.from_numpy(np.array(load("valid")))
    depths = 0
    while os.path.exists(os.path.join(args.cache, f"delta_{depths}.npy")):
        depths += 1
    cached = (valid > 0).nonzero()[:, 0]
    print(f"{args.cache}: {meta['N']} {dataset} training images of {meta['D']} pixels, {depths} depths on disk; "
          f"{len(cached)} images cached, to depth {int(valid.max())} at most (median "
          f"{int(valid[cached].median()) if len(cached) else 0})")
    if args.images is not None:
        idx = torch.tensor(args.images)
    else:
        gen = torch.Generator().manual_seed(args.seed)
        idx = cached[torch.randperm(len(cached), generator=gen)[:args.n]].sort().values
    if not len(idx) or not depths:
        print("nothing cached to draw")
        return

    X = pixels(load_images(dataset, "train", args.data_root)[:meta["N"]])[idx]
    deltas = torch.stack([torch.from_numpy(np.array(load(f"delta_{d}")[idx.numpy()])).float()
                          for d in range(depths)], 1)  # [n, depths, D]
    have = valid[idx]
    shown = columns(depths, args.cols - 1)
    recon = mean + deltas.cumsum(1)  # [n, depths, D]: after depth d's split
    blank = torch.ones_like(X)
    grid = torch.stack([X] + [torch.where((have >= k)[:, None], recon[:, k - 1], blank) for k in shown], 1)
    out = args.out or artifact("")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "cache_paths.png")
    save_grid(grid.clamp(0, 1).reshape(-1, *shape), path, nrow=len(shown) + 1)
    print(f"{path}: rows are training images {idx.tolist()} (cached to depths {have.tolist()}), then their "
          f"cached reconstruction after {shown} tokens")
    if args.deltas:
        scale = deltas.abs().amax((1, 2)).clamp_min(1e-8)[:, None]
        cells = [torch.where((have > k - 1)[:, None], 0.5 + deltas[:, k - 1] / (2 * scale), blank) for k in shown]
        path = os.path.join(out, "cache_deltas.png")
        save_grid(torch.stack([X] + cells, 1).clamp(0, 1).reshape(-1, *shape), path, nrow=len(shown) + 1)
        print(f"{path}: the same images, then the delta of their split at each of those depths (grey zero)")


if __name__ == "__main__":
    main()
