"""Decode a fixed random part of benjamin-paine/imagenet-1k-128x128 (ImageNet at 128 x 128; ImageNet's terms:
non-commercial research and education) into uint8 caches for dvae/data.py's "imagenet128": --train images drawn at
random from all of ImageNet's training set (its files are sorted by class), --val from its validation set (our val)
and --test from its test set, each [N, 3, 128, 128] with their labels (-1 where the test set has none).

The parquet files go in data/imagenet128/raw (data/imagenet128/raw/download.sh fetches them); the caches go in
data/imagenet128/{train,val,test}.pt.

Run from the repo: python prepare_imagenet128.py
"""

import argparse
import glob
import io
import os
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image


def decode(cell) -> np.ndarray:
    image = Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
    if image.size != (128, 128):
        image = image.resize((128, 128), Image.Resampling.BICUBIC)
    return np.asarray(image).transpose(2, 0, 1)


def take(files, count, seed, name, out):
    """count rows drawn at random (fixed by seed) from the parquet files, decoded, saved to out."""
    began = time.time()
    rows = [pq.ParquetFile(f).metadata.num_rows for f in files]
    total = sum(rows)
    chosen = np.sort(np.random.default_rng(seed).choice(total, size=min(count, total), replace=False))
    images = np.empty((len(chosen), 3, 128, 128), dtype=np.uint8)
    labels = np.full(len(chosen), -1, dtype=np.int64)
    done, start = 0, 0
    for f, n in zip(files, rows):
        here = chosen[(chosen >= start) & (chosen < start + n)] - start
        if len(here):
            table = pq.read_table(f).take(here)
            cells = table.column("image").to_pylist()
            got = table.column("label").to_pylist() if "label" in table.column_names else [None] * len(cells)
            for k, (cell, label) in enumerate(zip(cells, got)):
                images[done + k] = decode(cell)
                labels[done + k] = -1 if label is None else label
            done += len(here)
            print(f"{time.strftime('%H:%M:%S')} {name}: {done} of {len(chosen)} ({os.path.basename(f)}); "
                  f"{time.time() - began:.0f}s", flush=True)
        start += n
    perm = np.random.default_rng(seed + 1).permutation(len(chosen))  # (no class order within the cache either)
    torch.save(dict(images=torch.from_numpy(images[perm]), labels=torch.from_numpy(labels[perm]),
                    source=[os.path.basename(f) for f in files], rows=chosen[perm]), out)
    print(f"{time.strftime('%H:%M:%S')} {name}: {len(chosen)} images, {len(set(labels.tolist()) - {-1})} classes -> "
          f"{out} ({os.path.getsize(out) / 2 ** 30:.2f} GB)", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=int, default=100_000)
    p.add_argument("--val", type=int, default=10_000)
    p.add_argument("--test", type=int, default=10_000)
    p.add_argument("--root", default="data/imagenet128")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    raw = os.path.join(args.root, "raw")
    for split, source, count in (("val", "validation", args.val), ("test", "test", args.test),
                                 ("train", "train", args.train)):
        files = sorted(glob.glob(os.path.join(raw, f"{source}-*.parquet")))
        assert files, f"no {source} files in {raw}"
        take(files, count, args.seed, split, os.path.join(args.root, f"{split}.pt"))


if __name__ == "__main__":
    main()
