"""Dump a delta tree's codes for train_prior.py: for every train, val and test image, its sides (rare or not) at
every depth and the tree's split at each node on its path (the rare side's mean delta r and the ratio m), routed
by the tree's own Lloyd steps. Splits depend only on the code, never on which images reach a node, so images go
down in chunks. Each split's file also holds the digit labels and which checkpoint made it. Progress (each chunk,
timestamped) goes to run.log beside the files.

Run from the repo: python dump_codes.py --ckpt artifacts/<run>/model.pt   (files go to artifacts/<job>/)
"""

import argparse
import os
import time

import torch

from dvae.artifacts import artifact
from dvae.data import get_device, load_labels, load_mnist
from dvae.deltatree import descend, load_tree, pixels, root_frontier, tree_steps
from dvae.logs import RunLog


@torch.no_grad()
def codes(decoders, steps, mean, X, batch, tol, chunk, log):
    """X's sides rare [N, K] (bool), m [N, K] and r [N, K, D] (half precision) along each image's path."""
    rares, ms, rs = [], [], []
    began = time.time()
    for i, part in enumerate(X.split(chunk)):
        fr = root_frontier(part.to(mean.device), mean)
        for decoder, depth_steps in zip(decoders, steps):
            fr = descend(decoder, fr, depth_steps, batch, tol)
        node = fr.node.cpu()
        rare, m, delta = fr.sides.cpu()[node] > 0, fr.ms.cpu()[node].float(), fr.deltas.cpu()[node].float()
        rares.append(rare)
        ms.append(m.half())
        rs.append(torch.where(rare[..., None], delta, delta / -m[..., None]).half())  # delta = r or -m r
        log(f"  {min((i + 1) * chunk, len(X))} of {len(X)} images coded ({time.time() - began:.0f}s)")
    return torch.cat(rares), torch.cat(ms), torch.cat(rs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--chunk", type=int, default=10000, help="images sent down together")
    p.add_argument("--limit", type=int, default=None, help="images per split (default: all)")
    p.add_argument("--out", default=None, help="folder for the files (default: the run's artifacts folder)")
    p.add_argument("--device", default=None)
    p.add_argument("--data-root", default="data")
    a = p.parse_args()
    device = get_device(a.device)
    ckpt = torch.load(a.ckpt, map_location=device, weights_only=False)
    decoders, args = load_tree(ckpt, device), ckpt["args"]
    steps, batch, tol = tree_steps(ckpt, len(decoders)), args["node_batch"], args.get("tol", 0.0)
    mean = ckpt["mean"].to(device)
    out = a.out or artifact("")
    os.makedirs(out, exist_ok=True)
    log = RunLog(os.path.join(out, "run.log"), device)
    log(f"=== dump_codes.py {vars(a)}: {len(decoders)} tokens")
    for split in a.splits:
        began = time.time()
        X = pixels(load_mnist(split, a.data_root)[:a.limit])
        log(f"{split}: {len(X)} images")
        rare, m, r = codes(decoders, steps, mean, X, batch, tol, a.chunk, log)
        path = os.path.join(out, f"{split}.pt")
        torch.save(dict(rare=rare, m=m, r=r, labels=load_labels(split, a.data_root)[:a.limit], ckpt=a.ckpt,
                        epoch=ckpt.get("epoch"), depths=len(decoders)), path)
        log(f"{path}: {len(X)} codes of {len(decoders)} tokens, {rare.float().mean():.3f} of tokens rare "
            f"({time.time() - began:.0f}s)")


if __name__ == "__main__":
    main()
