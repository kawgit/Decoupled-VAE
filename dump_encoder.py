"""Stage 2: dump the encoder's codes, for the prior (train_prior.py) and the decoder (train_decoder.py). For every
train, val and test image: its token at every depth, and at each node on its path the encoder's children (their
centroids as offsets from the node's predicted mean, and the shares the encoder implies for them), routed by the
encoder's own Lloyd steps. The checkpoint may be train_encoder.py's hyperplane encoder (dvae/deltatree.py: two
children, token 1 the rare side, its offset r and the common side's -m r, implied shares m / (1 + m) and
1 / (1 + m)) or train_refiner.py's refining one (dvae/refine.py). Splits depend only on the code, never on which images reach a node, so images go down in
chunks. Each split's file also holds the digit labels, the training mean (a node's predicted mean is it plus the
offsets of the children taken so far) and which checkpoint made it (dvae/codes.py's TokenCodes). Progress (each
chunk, timestamped) goes to run.log beside the files.

Run from the repo: python dump_encoder.py --ckpt artifacts/<run>/model.pt   (files go to artifacts/<job>/)
"""

import argparse
import os
import time

import torch

from dvae.artifacts import artifact
from dvae.data import get_device, load_labels, load_mnist
from dvae.deltatree import descend, load_tree, pixels, tree_steps
from dvae.deltatree import root_frontier as tree_root
from dvae.logs import RunLog
from dvae.refine import Refiner, children, choose, load_refiner, root_frontier


@torch.no_grad()
def tree_codes(encoders, steps, mean, X, batch, tol, chunk, log):
    """codes for a hyperplane encoder (dvae/deltatree.py's, one per depth or shared): token 1 the rare side."""
    tokens, shares, offsets = [], [], []
    began = time.time()
    for i, part in enumerate(X.split(chunk)):
        fr = tree_root(part.to(mean.device), mean)
        for encoder, depth_steps in zip(encoders, steps):
            fr = descend(encoder, fr, depth_steps, batch, tol)
        node = fr.node.cpu()
        rare, m, delta = fr.sides.cpu()[node] > 0, fr.ms.cpu()[node].float(), fr.deltas.cpu()[node].float()
        r = torch.where(rare[..., None], delta, delta / -m[..., None])  # delta = r or -m r
        tokens.append(rare.long())
        shares.append(torch.stack([1 / (1 + m), m / (1 + m)], -1))
        offsets.append(torch.stack([-m[..., None] * r, r], 2).half())
        log(f"  {min((i + 1) * chunk, len(X))} of {len(X)} images coded ({time.time() - began:.0f}s)")
    return torch.cat(tokens), torch.cat(shares), torch.cat(offsets)


@torch.no_grad()
def codes(encoder: Refiner, mean, X, batch, chunk, log):
    """X's tokens [N, K], every child's share [N, K, V] and centroid offset [N, K, V, D] (half precision) along
    each image's path."""
    tokens, shares, offsets = [], [], []
    began = time.time()
    for i, part in enumerate(X.split(chunk)):
        fr = root_frontier(part.to(mean.device), mean)
        tok, w, cent = [], [], []
        for _ in range(encoder.depths):
            s = choose(encoder, fr, batch)
            child, t, _ = children(fr, s, encoder.V)
            tok.append(t.cpu())
            w.append(s.w[fr.node].cpu())
            cent.append(s.cent[fr.node].half().cpu())
            fr = child
        tokens.append(torch.stack(tok, 1))
        shares.append(torch.stack(w, 1))
        offsets.append(torch.stack(cent, 1))
        log(f"  {min((i + 1) * chunk, len(X))} of {len(X)} images coded ({time.time() - began:.0f}s)")
    return torch.cat(tokens), torch.cat(shares), torch.cat(offsets)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="train_encoder.py's or train_refiner.py's checkpoint")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--chunk", type=int, default=10000, help="images sent down together")
    p.add_argument("--limit", type=int, default=None, help="images per split (default: all)")
    p.add_argument("--out", default=None, help="folder for the files (default: the run's artifacts folder)")
    p.add_argument("--device", default=None)
    p.add_argument("--data-root", default="data")
    a = p.parse_args()
    device = get_device(a.device)
    ckpt = torch.load(a.ckpt, map_location=device, weights_only=False)
    args, mean = ckpt["args"], ckpt["mean"].to(device)
    refining = "V" in ckpt.get("encoder", {}).get("config", {})
    if refining:
        encoder = load_refiner(ckpt, device)
        K, V = encoder.depths, encoder.V

        def dump(X):
            return codes(encoder, mean, X, args["node_batch"], a.chunk, log)
    else:
        encoders = load_tree(ckpt, device)
        steps, K, V = tree_steps(ckpt, len(encoders)), len(encoders), 2

        def dump(X):
            return tree_codes(encoders, steps, mean, X, args["node_batch"], args.get("tol", 0.0), a.chunk, log)
    out = a.out or artifact("")
    os.makedirs(out, exist_ok=True)
    log = RunLog(os.path.join(out, "run.log"), device)
    log(f"=== dump_encoder.py {vars(a)}: {'refining' if refining else 'hyperplane'} encoder, {K} tokens, V = {V}, "
        f"from epoch {ckpt.get('epoch')}")
    for split in a.splits:
        began = time.time()
        X = pixels(load_mnist(split, a.data_root)[:a.limit])
        log(f"{split}: {len(X)} images")
        tokens, shares, offsets = dump(X)
        path = os.path.join(out, f"{split}.pt")
        torch.save(dict(tokens=tokens, shares=shares, offsets=offsets, labels=load_labels(split, a.data_root)[:a.limit],
                        mean=mean.cpu(), ckpt=a.ckpt, epoch=ckpt.get("epoch"), depths=K, V=V),
                   path)
        bits = -shares.gather(2, tokens[..., None])[..., 0].log2().sum(1).mean()
        log(f"{path}: {len(X)} codes of {K} tokens, {bits:.3f} bits each under the encoder's implied shares "
            f"({time.time() - began:.0f}s)")


if __name__ == "__main__":
    main()
