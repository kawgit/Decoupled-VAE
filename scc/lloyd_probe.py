"""How much do a shared decoder's Lloyd steps limit its splits? Routes the val images down a train_walk.py
checkpoint's tree (its own steps), then at every depth runs the decoder's Lloyd iterations for --most steps and
reports the val mse after that depth's split had it stopped after 1, 2, 4, ... steps, or at the step whose split
the decoder predicts is best (per image, m |r|^2: the predicted side means' gap, weighted by how evenly they
split). Then re-routes the whole tree with each step count and reports the val mse after every token.

Run from the repo: python scc/lloyd_probe.py --ckpt artifacts/<run>/model.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from dvae.artifacts import artifact  # noqa: E402
from dvae.data import get_device, load_mnist  # noqa: E402
from dvae.deltatree import D, lloyd, load_tree, pixels, root_frontier, split, tree_steps  # noqa: E402


@torch.no_grad()
def probe_depth(decoder, fr, most, batch):
    """Per Lloyd step t < most: the squared error of fr's images after a split at each node's step-t hyperplane
    [most], and after a split at the step (up to t) with the largest predicted gain [most]."""
    device = fr.R.device
    sse, best_sse = torch.zeros(most, device=device), torch.zeros(most, device=device)
    for i in range(0, len(fr.bits), batch):
        nodes = torch.arange(i, min(i + batch, len(fr.bits)), device=device)
        mine = (fr.node >= i) & (fr.node < i + len(nodes))
        R, node = fr.R[mine], fr.node[mine] - i
        mems = decoder.memory(*fr.history(nodes, device))
        best_gain = torch.full((len(nodes),), -1.0, device=device)
        best_err = torch.zeros(len(R), device=device)
        for t, (_, n, b, r, m, _) in enumerate(lloyd(decoder, mems, fr.starts(decoder, nodes), most)):
            rare = (R * n[node]).sum(-1) >= b[node]
            err = (R - torch.where(rare[:, None], r[node], -m[node, None] * r[node])).square().sum(-1)
            sse[t] += err.sum()
            gain = m * r.square().sum(-1)
            better = gain > best_gain
            best_gain = torch.where(better, gain, best_gain)
            best_err = torch.where(better[node], err, best_err)
            best_sse[t] += best_err.sum()
    return sse, best_sse


@torch.no_grad()
def route(decoders, X, mean, steps, batch, tol):
    """The val mse after every token, the tree built with steps Lloyd steps per node."""
    fr, mses = root_frontier(X, mean, resident=True), []
    for decoder in decoders:
        fr, _, _ = split(decoder, fr, steps, batch, tol)
        mses.append(fr.R.square().mean().item())
    return mses


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--most", type=int, default=32, help="Lloyd steps run at each depth")
    p.add_argument("--batch", type=int, default=512, help="nodes at once")
    p.add_argument("--device", default=None)
    p.add_argument("--data-root", default="data")
    a = p.parse_args()
    device = get_device(a.device)
    ckpt = torch.load(a.ckpt, map_location=device, weights_only=False)
    decoders, args = load_tree(ckpt, device), ckpt["args"]
    for d in decoders:
        d.eval()
    trained, tol = tree_steps(ckpt, len(decoders)), args.get("tol", 0.0)
    X = pixels(load_mnist("val", a.data_root)).to(device)
    mean = ckpt["mean"].to(device)
    out = open(artifact("probe.txt"), "a")

    def report(text):
        print(text, flush=True)
        out.write(text + "\n")
        out.flush()

    shown = [t for t in (1, 2, 4, 8, 16, 32, 64) if t <= a.most]
    report(f"{a.ckpt}: trained with {trained} Lloyd steps per depth; val mse after each depth's split (on the tree as "
           f"trained) if every node there stopped after s steps, or at its best predicted step among the first s")
    report("depth  before   " + "  ".join(f"s={s:<5d}" for s in shown) + "  | best of " +
           "  ".join(f"{s:<6d}" for s in shown))
    fr = root_frontier(X, mean, resident=True)
    N = len(X) * D
    totals = torch.zeros(2, a.most, device=device)
    for depth, decoder in enumerate(decoders):
        sse, best = probe_depth(decoder, fr, a.most, a.batch)
        totals += torch.stack([sse, best]) - fr.R.square().sum()
        report(f"{depth:5d}  {fr.R.square().sum().item() / N:.5f}  "
               + "  ".join(f"{sse[s - 1].item() / N:.5f}" for s in shown) + "  | "
               + "  ".join(f"{best[s - 1].item() / N:.5f}" for s in shown))
        fr, _, _ = split(decoder, fr, trained[depth], a.batch, tol)
    report("mean change per depth: " + "  ".join(f"s={s} {totals[0, s - 1].item() / N / len(decoders):+.6f}"
                                                 for s in shown)
           + " | best of: " + "  ".join(f"{s} {totals[1, s - 1].item() / N / len(decoders):+.6f}" for s in shown))
    report("whole tree re-routed with s steps everywhere: val mse after each token")
    for s in shown:
        mses = route(decoders, X, mean, s, a.batch, tol)
        report(f"  s={s:<3d} after {len(mses)} tokens {mses[-1]:.5f}; every 4th token: "
               + " ".join(f"{m:.5f}" for m in mses[3::4]))


if __name__ == "__main__":
    main()
