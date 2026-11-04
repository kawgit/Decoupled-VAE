"""Train a categorical leaf decoder for a delta tree (dvae/deltatree.py's LeafDecoder): p(x | leaf), 256 grey
levels per pixel, by maximum likelihood on the training images at their leaves after --tokens tokens.

The tree's decoders route every split (train, val, test) down to its leaves and price each image's code under
their own shares (the rare side's p = m / (1 + m)). The leaf decoder reads a leaf's delta history and its
predicted mean image. Early stopping watches the held-out (val) images' bits per pixel under it. Then the
test set is scored as -log2 p(code) - log2 p(x | code), the one-code upper bound on -log2 p(x).
"""

import argparse
import json
import math
import os
import time

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")  # ~6.5 GB: fail instead of swapping
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dvae.artifacts import artifact  # noqa: E402
from dvae.data import get_device, load_mnist  # noqa: E402
from dvae.logs import RunLog  # noqa: E402
from dvae.deltatree import D, LeafDecoder, load_tree, pixels, root_frontier, split  # noqa: E402


def leaf_bits(decoder, fr, X, levels, idx):
    """-log2 p(x | leaf) for images idx of a split [b]."""
    device = X.device
    logits = decoder(*fr.history(fr.node[idx], device), X[idx] - fr.R[idx])
    return F.cross_entropy(logits, levels[idx], reduction="none").sum(1) / math.log(2)


@torch.no_grad()
def evaluate(decoder, fr, X, levels, batch):
    """-log2 p(x | leaf) for every image [N]."""
    decoder.eval()
    idx = torch.arange(len(X), device=X.device)
    return torch.cat([leaf_bits(decoder, fr, X, levels, idx[i:i + batch]) for i in range(0, len(X), batch)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="checkpoints/deltatree.pt", help="the tree (train.py)")
    p.add_argument("--tokens", type=int, default=None, help="code length (default: all the tree's depths)")
    p.add_argument("--out", default=None,
                   help="leaf decoder checkpoint (default: leaf.pt in the run's artifacts folder)")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--channels", type=int, default=64, help="width of the mean image's convolutions")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--limit-test", type=int, default=None)
    p.add_argument("--results", default=None, help="results prefix (default: results in the run's artifacts folder)")
    p.add_argument("--log", default=None, help="summary log (default: run.log in the run's artifacts folder)")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="data")
    args = p.parse_args()
    device = get_device(args.device)
    out, results = args.out or artifact("leaf.pt"), args.results or artifact("results")
    log = RunLog(args.log or artifact("run.log"), device)

    log(f"=== {time.strftime('%Y-%m-%d')} {vars(args)}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    a, mean = ckpt["args"], ckpt["mean"].to(device)
    decoders = load_tree(ckpt, device)
    K = args.tokens if args.tokens is not None else len(decoders)
    steps, batch = a["lloyd_steps"], a["node_batch"]
    images = {"train": load_mnist("train", args.data_root)[:a["limit_train"]],
              "val": load_mnist("val", args.data_root)[:a["limit_val"]],
              "test": load_mnist("test", args.data_root)[:args.limit_test]}
    X = {s: pixels(im).to(device) for s, im in images.items()}
    levels = {s: im.long().flatten(1).to(device) for s, im in images.items()}
    fr = {s: root_frontier(x, mean) for s, x in X.items()}
    code = {s: torch.zeros(len(x), device=device) for s, x in X.items()}
    for k in range(K):
        tree = decoders[k]
        for s in fr:
            fr[s], token_bits, _ = split(tree, fr[s], steps, batch, a.get("tol", 0.0))
            code[s] += token_bits
    for s in fr:
        log(f"{s}: {len(X[s])} images in {len(fr[s].bits)} leaves after {K} tokens, "
            f"code {code[s].mean().item():.2f} bits, mse {fr[s].R.square().mean().item():.5f}")

    torch.manual_seed(args.seed)
    decoder = LeafDecoder(K, args.width, args.layers, args.heads, args.channels).to(device)
    log(f"leaf decoder: {sum(q.numel() for q in decoder.parameters()) / 1e6:.1f}M parameters")
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gen = torch.Generator().manual_seed(args.seed)
    best: tuple[float, int, dict | None] = (math.inf, -1, None)
    N = len(X["train"])
    for epoch in range(args.epochs):
        t0 = time.time()
        decoder.train()
        total = 0.0
        order = torch.randperm(N, generator=gen).to(device)
        for i in range(0, N, args.batch):
            bits = leaf_bits(decoder, fr["train"], X["train"], levels["train"], order[i:i + args.batch])
            loss = bits.mean() / D
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            opt.step()
            total += bits.sum().item()
        t1 = time.time()
        val = evaluate(decoder, fr["val"], X["val"], levels["val"], args.batch).mean().item() / D
        improved = val < best[0]
        log(f"  epoch {epoch:2d}: val {val:.4f} pixel bits/dim (with the code "
            f"{val + code['val'].mean().item() / D:.4f}); train {total / (N * D):.4f} "
            f"({t1 - t0:.0f}s train, {time.time() - t1:.0f}s val){'  *' if improved else ''}")
        if improved:
            best = (val, epoch, {key: v.detach().clone() for key, v in decoder.state_dict().items()})
            torch.save(dict(config=decoder.config, state_dict=best[2], tree=args.ckpt, tokens=K, args=vars(args)), out)
        elif epoch - best[1] >= args.patience:
            break
    assert best[2] is not None
    decoder.load_state_dict(best[2])
    log(f"kept epoch {best[1]} ({best[0]:.4f} val pixel bits/dim), saved {out}")

    rows = {}
    for s in ("val", "test"):
        leaf = evaluate(decoder, fr[s], X[s], levels[s], args.batch)
        total = code[s] + leaf
        rows[s] = dict(images=len(total), tokens=K, bpd=total.mean().item() / D,
                       nats=total.mean().item() * math.log(2), code_bits=code[s].mean().item(),
                       pixel_bpd=leaf.mean().item() / D)
        r = rows[s]
        log(f"{s}: {r['bpd']:.4f} bits/dim ({r['nats']:.1f} nats/image): code {r['code_bits']:.2f} bits, "
            f"pixels {r['pixel_bpd']:.4f} bits/dim, over {r['images']} images")
    os.makedirs(os.path.dirname(results) or ".", exist_ok=True)
    with open(results + ".json", "w") as f:
        json.dump(dict(tree=args.ckpt, leaf=out, bound="one code: -log2 p(z(x)) - log2 p(x | z(x))", **rows), f, indent=2)
    with open(results + ".md", "w") as f:
        f.write("| split | tokens | bpd | nats/image | code bits | pixel bpd |\n|---|---|---|---|---|---|\n")
        for s, r in rows.items():
            f.write(f"| {s} | {r['tokens']} | {r['bpd']:.4f} | {r['nats']:.1f} | {r['code_bits']:.2f} | "
                    f"{r['pixel_bpd']:.4f} |\n")
    log(f"wrote {results}.json and .md")


if __name__ == "__main__":
    main()
