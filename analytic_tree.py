"""The val mse after each token of an analytic encoder fit on images from a train_rounds.py generator, at several
sample sizes: how well images drawn from the generator would teach an encoder that knows its data exactly.

The analytic encoder is train_encoder.py's delta tree with the network replaced by the truth: at every node,
Lloyd's algorithm for two means runs on the node's own fit images (the side means it uses are the images'
actual means), starting from the depth's top principal direction through the node's mean, for --steps steps;
the rare side is the smaller one. The last hyperplane splits the node and each child's delta is its side's
mean. A node that cannot split (under two images, or one side empty) keeps its images on its common side with
no delta. The real val images (and --held-out generated ones not in any fit) go down the same hyperplanes, each
moved by its side's delta, and their mse is logged after every token.

The generated images are drawn once (--sizes' largest), the fits use nested prefixes of them in increasing size,
and each size's curves are logged as soon as its fit ends. With --real, the analytic encoder fit on the real
training images comes first, as a baseline. Results go to analytic.json and analytic.png in the run's folder.

Run from the repo: python analytic_tree.py --generator artifacts/429-gen-lloyd16-from386/round_0.pt
"""

import argparse
import json
import os
import time

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")  # ~6.5 GB: fail instead of swapping
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from dvae.artifacts import artifact  # noqa: E402
from dvae.data import get_device, load_mnist  # noqa: E402
from dvae.deltatree import D, pixels  # noqa: E402
from dvae.logs import RunLog  # noqa: E402
from dvae.rounds import CodeEncoder, CodePrior, Decoder, DecoderCov, DecoderMean, LloydEncoder, sample_codes  # noqa: E402


def load_generator(path, device):
    """train_rounds.py's round_<r>.pt as (encoder, prior, decoder, mean, K, V), in eval mode."""
    ckpt = torch.load(path, map_location=device, weights_only=False)

    def restore(cls, entry):
        model = cls(**entry["config"]).to(device)
        model.load_state_dict(entry["state_dict"])
        return model.eval()

    encoder = restore(LloydEncoder if ckpt["encoder"]["kind"] == "lloyd" else CodeEncoder, ckpt["encoder"])
    prior = restore(CodePrior, ckpt["prior"])
    cov = restore(DecoderCov, ckpt["decoder_cov"]) if ckpt.get("decoder_cov") else None
    decoder = Decoder(restore(DecoderMean, ckpt["decoder_mean"]), cov).eval()
    return encoder, prior, decoder, ckpt["mean"].to(device), int(ckpt["args"]["depths"]), int(ckpt["args"]["V"])


def gram_top(R: Tensor, chunk: int) -> Tensor:
    """The top principal direction of the rows of R [N, D] (any dtype), about 0, a unit float vector [D]."""
    gram = torch.zeros(D, D, dtype=torch.float64, device=R.device)
    for i in range(0, len(R), chunk):
        x = R[i:i + chunk].float()
        gram += (x.T @ x).double()
    return torch.linalg.eigh(gram)[1][:, -1].float()


class Routed:
    """Images routed down the tree beside the fit: their remainders R [n, D] (float) and nodes [n]."""

    def __init__(self, X: Tensor, mean: Tensor):
        self.R = X.float() - mean
        self.node = torch.zeros(len(X), dtype=torch.long, device=X.device)

    def mse(self) -> float:
        return self.R.square().sum().item() / self.R.numel()


@torch.no_grad()
def fit(R: Tensor, others: list[Routed], depths: int, steps: int, chunk: int, log) -> tuple[list[float], list]:
    """Fits the analytic tree on remainders R [N, D] (each image minus the fit images' mean; half precision, updated
    in place to the leaves' remainders), routing others alongside. Returns the fit images' mse after each token,
    and each of others' (one list per routed set)."""
    N, device = len(R), R.device
    node, n_nodes = torch.zeros(N, dtype=torch.long, device=device), 1
    fit_mse, other_mse = [], [[] for _ in others]
    for depth in range(depths):
        began = time.time()
        u = gram_top(R, chunk)
        order = torch.argsort(node, stable=True)
        counts = torch.bincount(node, minlength=n_nodes)
        first = torch.cumsum(counts, 0) - counts  # each node's first index into order
        # groups: the nodes whose first image falls in the same chunk of order, so at most chunk + 1 nodes each
        group = (first // chunk).cpu()
        bounds = torch.unique_consecutive(group, return_counts=True)[1].cumsum(0).tolist()
        side = torch.zeros(N, dtype=torch.bool, device=device)
        other_side = [torch.zeros(len(o.R), dtype=torch.bool, device=device) for o in others]
        other_order = [torch.argsort(o.node, stable=True) for o in others]
        edges = torch.tensor([0] + bounds, device=device)
        other_cuts = [torch.searchsorted(o.node[oo], edges).tolist() for o, oo in zip(others, other_order)]
        sse, unsplit, lo = 0.0, 0, 0
        for g, hi in enumerate(bounds):  # nodes lo..hi-1
            G = hi - lo
            s, e = int(first[lo]), int(first[hi - 1] + counts[hi - 1])
            pieces = [(i, min(i + chunk, e)) for i in range(s, e, chunk)]

            all_n = counts[lo:hi].float()
            all_sum = torch.zeros(G, D, device=device)
            for i, j in pieces:
                idx = order[i:j]
                all_sum.index_add_(0, node[idx] - lo, R[idx].float())
            def measure(n, b, all_n=all_n, all_sum=all_sum, pieces=pieces, lo=lo):
                """The sides of hyperplanes (n, b), turned so the rare side is the smaller: (n, b, the rare
                side's count, its mean, the common side's mean)."""
                rare_n, rare_sum = torch.zeros(len(n), device=device), torch.zeros(len(n), D, device=device)
                for i, j in pieces:
                    idx = order[i:j]
                    x, ln = R[idx].float(), node[idx] - lo
                    rare = (x * n[ln]).sum(-1) >= b[ln]
                    rare_n.index_add_(0, ln, rare.float())
                    rare_sum.index_add_(0, ln[rare], x[rare])
                common_n, common_sum = all_n - rare_n, all_sum - rare_sum
                flip = rare_n > common_n
                n, b = torch.where(flip[:, None], -n, n), torch.where(flip, -b, b)
                rare_n, common_n = torch.where(flip, common_n, rare_n), torch.where(flip, rare_n, common_n)
                rare_sum, common_sum = (torch.where(flip[:, None], common_sum, rare_sum),
                                        torch.where(flip[:, None], rare_sum, common_sum))
                return (n, b, rare_n, rare_sum / rare_n.clamp_min(1)[:, None],
                        common_sum / common_n.clamp_min(1)[:, None])

            n, b, rare_n, rare_mean, common_mean = measure(u.expand(G, D), (all_sum @ u) / all_n.clamp_min(1))
            for _ in range(steps - 1):  # each next hyperplane: the bisector of the last side means
                normal = rare_mean - common_mean
                normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                moving = rare_n > 0
                n = torch.where(moving[:, None], normal, n)
                b = torch.where(moving, (normal * (rare_mean + common_mean)).sum(-1) / 2, b)
                n, b, rare_n, rare_mean, common_mean = measure(n, b)
            splits = rare_n > 0  # (so common_n > 0 too)
            unsplit += int((~splits).sum())
            # an unsplit node: every image on its common side, with no delta
            rare_mean[~splits] = 0
            common_mean[~splits] = 0
            b = torch.where(splits, b, torch.full_like(b, float("inf")))
            for i, j in pieces:
                idx = order[i:j]
                x, ln = R[idx].float(), node[idx] - lo
                rare = (x * n[ln]).sum(-1) >= b[ln]
                x = x - torch.where(rare[:, None], rare_mean[ln], common_mean[ln])
                R[idx] = x.to(R.dtype)
                side[idx] = rare
                sse += x.square().sum().item()
            for o, oo, cuts, os_ in zip(others, other_order, other_cuts, other_side):
                idx = oo[cuts[g]:cuts[g + 1]]
                ln = o.node[idx] - lo
                x = o.R[idx]
                rare = (x * n[ln]).sum(-1) >= b[ln]
                o.R[idx] = x - torch.where(rare[:, None], rare_mean[ln], common_mean[ln])
                os_[idx] = rare
            lo = hi
        keys, node = torch.unique(node * 2 + side, return_inverse=True)
        n_nodes = len(keys)
        for o, os_ in zip(others, other_side):
            okey = o.node * 2 + os_
            o.node = torch.searchsorted(keys, okey)
            assert bool((keys[o.node.clamp_max(n_nodes - 1)] == okey).all()), "a routed image left the fit's nodes"
        fit_mse.append(sse / (N * D))
        for m, o in zip(other_mse, others):
            m.append(o.mse())
        log.detail(f"  depth {depth + 1}: {n_nodes} nodes ({unsplit} of the parents unsplit), fit mse "
                   f"{fit_mse[-1]:.6f}, " + ", ".join(f"{m[-1]:.6f}" for m in other_mse)
                   + f" ({time.time() - began:.0f}s)")
    return fit_mse, other_mse


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generator", required=True, help="train_rounds.py's round_<r>.pt")
    p.add_argument("--sizes", type=int, nargs="+", default=[50000, 200000, 1000000, 4000000, 8000000])
    p.add_argument("--steps", type=int, default=16, help="Lloyd steps per node")
    p.add_argument("--depths", type=int, default=None, help="tokens (default: the generator's)")
    p.add_argument("--held-out", type=int, default=50000, help="generated images routed but never fit")
    p.add_argument("--real", action="store_true", help="first fit the real training images, as a baseline")
    p.add_argument("--piece", type=int, default=65536,
                   help="images drawn per sample_codes call (its Draw keeps every depth's centroids)")
    p.add_argument("--sample-batch", type=int, default=32768)
    p.add_argument("--chunk", type=int, default=2 ** 18, help="images per step of the fit's loops")
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--data-root", default="data")
    args = p.parse_args()
    device = get_device(args.device)
    log = RunLog(artifact("run.log"), device)
    log(f"=== {time.strftime('%Y-%m-%d')} analytic_tree.py {vars(args)} on {device}")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    encoder, prior, decoder, gen_mean, K, V = load_generator(args.generator, device)
    depths = args.depths or K
    assert V == 2, "the analytic tree is binary"
    Xv = pixels(load_mnist("val", args.data_root)[:args.limit_val]).to(device)
    sizes = sorted(args.sizes)

    def draw(n):
        out = torch.empty(n, D, dtype=torch.half, device=device)
        began = time.time()
        for i in range(0, n, args.piece):
            m = min(args.piece, n - i)
            out[i:i + m] = sample_codes(encoder, prior, decoder, gen_mean, m, K, V, args.sample_batch).leaf.half()
            log.detail(f"  drew {i + m} of {n} images ({time.time() - began:.0f}s)")
        return out

    held = draw(args.held_out) if args.held_out else None
    curves: list[dict] = []

    def run_fit(label, X):
        began = time.time()
        mean = X.float().mean(0) if len(X) <= args.chunk else \
            sum(X[i:i + args.chunk].float().sum(0) for i in range(0, len(X), args.chunk)) / len(X)
        assert isinstance(mean, Tensor)
        R = torch.empty_like(X)
        for i in range(0, len(X), args.chunk):
            R[i:i + args.chunk] = (X[i:i + args.chunk].float() - mean).half()
        others = [Routed(Xv, mean)] + ([Routed(held, mean)] if held is not None else [])
        fit_mse, other = fit(R, others, depths, args.steps, args.chunk, log)
        del R
        val = other[0]
        entry = dict(label=label, n=len(X), fit=fit_mse, val=val, held_out=other[1] if len(other) > 1 else None,
                     seconds=time.time() - began)
        curves.append(entry)
        log(f"{label}: val mse after {depths} tokens {val[-1]:.6f} (per depth " + " ".join(f"{m:.5f}" for m in val)
            + f"); fit mse {fit_mse[-1]:.6f}"
            + (f", held-out generated {entry['held_out'][-1]:.6f}" if entry["held_out"] else "")
            + f" ({entry['seconds']:.0f}s)")
        with open(artifact("analytic.json"), "w") as f:
            json.dump(dict(args={k: str(v) for k, v in vars(args).items()}, fits=curves), f, indent=1)
        plot(curves, depths)

    if args.real:
        X = pixels(load_mnist("train", args.data_root)).to(device).half()
        run_fit(f"real train ({len(X)})", X)
        del X
    G, have = torch.empty(sizes[-1], D, dtype=torch.half, device=device), 0
    for n in sizes:
        log(f"drawing {n - have} more images from {args.generator}")
        G[have:n] = draw(n - have)
        have = n
        run_fit(f"generated {n}", G[:n])
    log("done")


def plot(curves, depths):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    tokens = list(range(1, depths + 1))
    for c in curves:
        style = dict(marker="o", markersize=3) | (dict(ls="--", color="k") if c["label"].startswith("real") else {})
        axes[0].plot(tokens, c["val"], label=c["label"], **style)
        axes[1].plot(tokens, c["fit"], label=c["label"], **style)
    for ax, title in zip(axes, ("val mse (real val images)", "fit mse (the images it was fit on)")):
        ax.set_yscale("log")
        ax.set_xlabel("tokens")
        ax.set_title(f"Analytic encoder: {title}")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    fig.tight_layout()
    fig.savefig(artifact("analytic.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
