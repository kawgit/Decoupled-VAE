"""Where a V-way tree's codes (train_inner_vway.py) keep their images: real test codes with only their first k tokens
kept and the rest drawn at random, uniformly or from the tree's shares, each replayed through the tree
(dvae/vreplay.py's VTree.sample) and decoded by the outer decoder. The smallest k whose images still look like the
originals says which depths carry what a prior has to learn.

What the run keeps (artifacts/<job>/): run.log; grid_uniform.png and grid_shares.png (a row per test image: the
original, its reconstruction by every depth, then the mixed codes for each k); mse.png (latent and pixel mse to the
originals against k, for both fills); records.json (those numbers).

Run from the repo: python mixed_codes.py --tree artifacts/<train_inner_vway.py job>
"""

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from dvae.artifacts import artifact  # noqa: E402
from dvae.data import get_device, load_images  # noqa: E402
from dvae.logs import RunLog  # noqa: E402
from dvae.outer import load_outer  # noqa: E402
from dvae.plots import COMPONENT_COLORS, _style  # noqa: E402
from dvae.viz import save_grid  # noqa: E402
from dvae.vreplay import VTree  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tree", required=True, metavar="RUN", help="a train_inner_vway.py run")
    p.add_argument("--depths", type=int, default=None, help="depths of the tree used (default: all)")
    p.add_argument("--ks", default="0,1,2,4,8,16,32,64,128,256,384,512",
                   help="real tokens kept, comma-separated (those above the depths are dropped)")
    p.add_argument("--n", type=int, default=8, help="test images shown (picked at random)")
    p.add_argument("--n-mse", type=int, default=1000, help="test images the mse is measured on")
    p.add_argument("--fp16", action="store_true", help="float16 autocast (V100s)")
    p.add_argument("--node-batch", type=int, default=2048)
    p.add_argument("--data-root", default="data")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or os.path.dirname(artifact("run.log"))
    os.makedirs(out, exist_ok=True)
    device = get_device(args.device)
    log = RunLog(os.path.join(out, "run.log"), device)
    log(f"=== mixed_codes.py {time.strftime('%Y-%m-%d')} {vars(args)}")

    tree = VTree(args.tree, device, args.depths, fp16=args.fp16, node_batch=args.node_batch)
    K, V = tree.K, tree.args["V"]
    ks = [k for k in (int(s) for s in args.ks.split(",")) if k <= K]
    model, lat = load_outer(tree.args["outer"], device)
    C, h, w = lat["shape"]
    mean, std = lat["mean"].to(device)[None, :, None, None], lat["std"].to(device)[None, :, None, None]
    test = lat["test"]
    idx = torch.randperm(len(test), generator=torch.Generator().manual_seed(args.seed))[:max(args.n, args.n_mse)]
    Z = test[idx].float().flatten(1).to(device)
    images = (load_images(lat["dataset"], "test", args.data_root)[idx].to(device).float() + 0.5) / 256
    log(f"tree {args.tree}: {K} depths, V={V}; {len(idx)} test images of {len(test)} (the first {args.n} shown); "
        f"k in {ks}")

    @torch.no_grad()
    def decode(z):
        model.decoder.eval()
        with torch.autocast(device.type, dtype=torch.float16, enabled=args.fp16):
            return model.decoder(z.float().view(-1, C, h, w) / lat["scale"] * std + mean).float().clamp(0, 1)

    toks = []
    g, _ = tree.code(Z, K, visit=lambda d, c, t, s, q: toks.append(t.clone()))
    T = torch.stack(toks, 1).long()  # [n, K]
    recon = Z - g.R
    y = decode(recon)
    coded = dict(latent=g.R.square().mean().item(), pixel=(y - images).square().mean().item())
    fills: dict[str, list[dict[str, float]]] = {}
    log(f"coded: latent mse {coded['latent']:.6f}, pixel mse {coded['pixel']:.6f}")
    for fill in ("uniform", "shares"):
        gen = torch.Generator(device).manual_seed(args.seed + 1)
        cols, rows = [images[:args.n], y[:args.n]], []
        for k in ks:
            def choose(d, cands, shares, parts, k=k):
                if d < k:
                    return T[:, d]
                weights = torch.ones_like(shares, dtype=torch.float) if fill == "uniform" else \
                    shares.float().clamp_min(0) + 1e-12
                return torch.multinomial(weights, 1, generator=gen)[:, 0]
            _, z = tree.sample(len(Z), choose, K)
            yk = decode(z)
            row = dict(k=k, latent=(z - Z).square().mean().item(), pixel=(yk - images).square().mean().item(),
                       to_recon=(z - recon).square().mean().item())
            rows.append(row)
            cols.append(yk[:args.n])
            log(f"{fill} after k={k}: latent mse {row['latent']:.6f} (to the depth-{K} reconstruction "
                f"{row['to_recon']:.6f}), pixel mse {row['pixel']:.6f}")
        fills[fill] = rows
        grid = torch.stack(cols, 1).flatten(0, 1)  # (row-major: image by image)
        save_grid(grid, os.path.join(out, f"grid_{fill}.png"), nrow=len(cols), scale=2)
        log(f"grid_{fill}.png columns: original | depth {K} | " + " | ".join(f"k={k}" for k in ks))

    with open(os.path.join(out, "records.json"), "w") as f:
        json.dump(dict(args=vars(args), K=K, V=V, ks=ks, recon=coded, fills=fills), f, indent=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    xs = [max(k, 0.5) for k in ks]  # (k = 0 drawn at 0.5 on the log axis)
    for ax, key, name in ((axes[0], "latent", "latent mse"), (axes[1], "pixel", "pixel mse")):
        for (fill, rows), color in zip(fills.items(), (COMPONENT_COLORS["val"], COMPONENT_COLORS["train"])):
            ax.plot(xs, [r[key] for r in rows], color=color, linewidth=1.6, marker="o", markersize=3,
                    label=f"rest {fill}")
        ax.axhline(coded[key], color="#666666", linestyle=":", linewidth=1.2, label=f"all {K} real")
        ax.set_xscale("log", base=2)
        _style(ax, f"{name} to the originals", "real tokens kept, k (k = 0 at 0.5)", name)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "mse.png"), dpi=130)
    plt.close(fig)
    log(f"done; {out}")


if __name__ == "__main__":
    main()
