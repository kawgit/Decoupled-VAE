"""What decides a V-way tree's tokens (train_inner_vway.py), as a prior has to learn it: images are coded through the
tree (dvae/vreplay.py's VTree, its loop repeated here to see every step), and at each depth each image's token (the
cell of its remainder R among the last Lloyd step's queries q) is compared with the nearest to R of: q itself (the
definition, so this should be all), the candidates c a prior is shown (the cells' predicted means, the deltas taken),
and the depth's start (step 0's centroids, the same at every node). Also: how many Lloyd steps each depth kept (one:
q is the start, the same partition at every node of the depth).

What the run keeps (artifacts/<job>/): run.log; records.json (each depth's agreement rates).

Run from the repo: python check_code_labels.py --tree artifacts/<train_inner_vway.py job> --depths 512
"""

import argparse
import json
import os
import time

import torch

from dvae.artifacts import artifact
from dvae.components import root_group
from dvae.data import get_device, load_images
from dvae.logs import RunLog
from dvae.outer import load_outer
from dvae.vreplay import VTree
from dvae.vway import cells_of, split_rows, vanswers, vdescend


def nearest(R, cents):
    """Each image's nearest of its own centroids cents [n, V, D] to R [n, D]."""
    cents = cents.float()
    return (cents.square().sum(-1) - 2 * torch.einsum("nd,nvd->nv", R, cents)).argmin(-1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tree", required=True)
    p.add_argument("--depths", type=int, default=None)
    p.add_argument("--n", type=int, default=1000, help="training images coded")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or os.path.dirname(artifact("run.log"))
    os.makedirs(out, exist_ok=True)
    device = get_device(args.device)
    log = RunLog(os.path.join(out, "run.log"), device)
    log(f"=== check_code_labels.py {time.strftime('%Y-%m-%d')} {vars(args)}")
    tree = VTree(args.tree, device, args.depths, fp16=args.fp16)
    K = tree.K
    if tree.args.get("outer"):
        _, lat = load_outer(tree.args["outer"], device)
        X = lat["train"][:args.n].float().flatten(1).to(device)
    else:  # (a tree of the images themselves, as train_inner_vway.py scales them)
        X = load_images(tree.args["dataset"], "train", tree.args.get("data_root", "data"))[:args.n]
        X = (X.float().flatten(1) / 255).to(device)
    log(f"tree {args.tree}: {K} depths; steps kept per depth: "
        f"{ {s: tree.steps.count(s) for s in sorted(set(tree.steps))} }; {len(X)} training images")

    g = root_group("check", X, tree.mean, tree.window)
    rows = []
    with torch.no_grad():
        for d in range(K):
            start = tree.starts[d].to(tree.cdtype).expand(g.nodes, -1, -1)
            q, c, pr = start, start, None
            for t, state in enumerate(tree.nets[d]):
                tree.net.load_state_dict(state)
                c, pr = vanswers(tree.net, g, q, tree.node_batch)
                if t + 1 < len(tree.nets[d]):
                    q = c
            assert pr is not None
            cells = cells_of(g, q)
            R = g.R.float()
            row = dict(d=d, steps=tree.steps[d], q=(nearest(R, q[g.node]) == cells).float().mean().item(),
                       c=(nearest(R, c[g.node]) == cells).float().mean().item(),
                       start=(nearest(R, start[g.node]) == cells).float().mean().item(),
                       share_max=(pr[g.node].argmax(-1) == cells).float().mean().item())
            rows.append(row)
            if d < 8 or d % 32 == 31:
                log(f"d{d:03d} ({row['steps']} steps): token = nearest of q {row['q'] * 100:.2f}%, of the candidates "
                    f"c {row['c'] * 100:.2f}%, of the depth's start {row['start'] * 100:.2f}%; the largest share's "
                    f"{row['share_max'] * 100:.2f}%")
            vdescend(g, cells, split_rows(g, cells, c, pr), tree.budget, tree.bits)
    mean = {k: sum(r[k] for r in rows) / K for k in ("q", "c", "start", "share_max")}
    log(f"over {K} depths: token = nearest of q {mean['q'] * 100:.2f}%, of c {mean['c'] * 100:.2f}%, of the start "
        f"{mean['start'] * 100:.2f}%; the largest share's {mean['share_max'] * 100:.2f}% (chance "
        f"{100 / tree.args['V']:.1f}%)")
    with open(os.path.join(out, "records.json"), "w") as f:
        json.dump(dict(args=vars(args), by_depth=rows, mean=mean), f)
    log(f"done; {out}")


if __name__ == "__main__":
    main()
