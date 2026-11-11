"""QA for a train_encoder_components.py run: its saved networks replayed down the tree it kept, on every val image
and on some train and test images, plus codes sampled with the encoder's own shares (each side rare with
probability m / (1 + m)). Each depth asks about the start its step-0 network's checkpoint holds; for a depth whose
checkpoint has none (runs from before starts were saved), the start is recomputed as the run found it, the top
direction of all the training remainders there, so every training image is replayed down to the last such depth
(and the starts it recomputes are written into those checkpoints, so later replays are fast).

compress_train.png, compress_val.png, compress_test.png: rows are images, each followed by its node's predicted
mean (its reconstruction) after a spread of code lengths ending with the tree's last depth. samples.png: a
--grid x --grid square of sampled codes' predicted means after the last depth; samples_by_depth.png: --n sampled
codes after the same spread of lengths. mse.png: the replay's val mse (all val images) and test mse by depth
beside the run's records, which the val replay matches exactly on the run's GPU type and precision (train too, for
the depths replayed on every training image).

Run from the repo: python qa_components.py --run artifacts/<job>   (pictures go to artifacts/<qa job>/)
"""

import argparse
import json
import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from dvae.artifacts import artifact  # noqa: E402
from dvae.components import (StepNet, answers, descend, mse_now, next_hyperplane, orient, root_group,  # noqa: E402
                             top_direction)
from dvae.data import SHAPES, get_device, load_images  # noqa: E402
from dvae.deltatree import pixels  # noqa: E402
from dvae.viz import save_grid  # noqa: E402


def spread(K, n):
    """About n code lengths from 1 to K, spaced evenly in log, ending with K."""
    if K <= n:
        return list(range(1, K + 1))
    out = sorted({max(1, round(math.exp(math.log(K) * i / (n - 1)))) for i in range(n)})
    return out[:-1] + [K] if out[-1] != K else out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="the run's folder (records.json and models/)")
    p.add_argument("--depths", type=int, default=None, help="replay this many depths (default: the tree kept)")
    p.add_argument("-n", type=int, default=16, help="images (and sampled codes) shown compressed")
    p.add_argument("--score", type=int, default=2000, help="test images whose mse is measured at every depth")
    p.add_argument("--grid", type=int, default=12, help="side of the square of samples")
    p.add_argument("--cols", type=int, default=12, help="code lengths shown per image")
    p.add_argument("--amp", choices=("run", "none", "bf16", "fp16"), default="run",
                   help="the networks' precision (run: the run's own)")
    p.add_argument("--node-batch", type=int, default=1024)
    p.add_argument("--save-starts", action=argparse.BooleanOptionalAction, default=True,
                   help="write each start recomputed from the training images into its step-0 checkpoint, so the "
                        "run's checkpoints are complete and later replays fast")
    p.add_argument("--out", default=None, help="folder for the pictures (default: this job's artifacts folder)")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="data")
    args = p.parse_args()
    device = get_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    with open(os.path.join(args.run, "records.json")) as f:
        rec = json.load(f)
    a = rec["args"]
    K = args.depths or rec["kept"]
    steps = {r["d"]: r["steps"] for r in rec["depths"]}
    step_train = {(r["d"], r["t"]): r["train"] for r in rec["models"]}  # (each network's train mse after its split)
    dataset, shape = a["dataset"], SHAPES[a["dataset"]]
    amp = args.amp if args.amp != "run" else ("bf16" if a.get("bf16") else "fp16" if a.get("fp16") else "none")
    out = args.out or os.path.dirname(artifact("x"))
    os.makedirs(out, exist_ok=True)
    window, bits = a.get("history_window", 0), a.get("table_bits", 16)
    print(f"{args.run}: replaying {K} depths ({sum(steps[d] for d in range(K))} networks) under {amp}")

    def path_of(d, t):
        return os.path.join(args.run, "models", f"d{d:03d}_s{t:02d}.pt")

    # (mapped, not read: a checkpoint's start is a few kB beside megabytes of weights)
    starts = [torch.load(path_of(d, 0), map_location="cpu", weights_only=False, mmap=True).get("start")
              for d in range(K)]
    missing = [d for d in range(K) if starts[d] is None]
    last_missing = missing[-1] if missing else -1
    print(f"{K - len(missing)} of {K} depths' starts are in their checkpoints" +
          (f"; every training image is replayed through depth {last_missing} to recompute the rest" if missing else ""))

    X = {s: pixels(load_images(dataset, s, args.data_root)[:a.get(f"limit_{s}")]).to(device)
         for s in ("train", "val", "test")}  # (the run's own limits: its Lloyd starts are its training images')
    if a.get("flip"):  # (and its mirror images)
        X["train"] = torch.cat([X["train"], X["train"].view(-1, *shape).flip(-1).flatten(1)])
    mean = torch.tensor(rec["mean"], device=device)
    count = max(args.n, args.grid ** 2)
    inputs = {"train": X["train"][:args.n], "val": X["val"], "test": X["test"][:max(args.n, args.score)],
              "sample": mean.expand(count, -1)}
    groups = {s: root_group(s, x.clone() if s == "sample" else x, mean, window) for s, x in inputs.items()}
    full = root_group("train", X["train"], mean, window) if missing else None  # (every training image)
    lengths = spread(K, args.cols)
    recon = {s: [inputs[s][:args.n]] if s != "sample" else [] for s in groups}  # (images first, then lengths)
    mse: dict[str, list[float]] = {"train": [], "val": [], "test": []}
    gen = torch.Generator(device=device).manual_seed(args.seed)
    nets: dict[tuple, StepNet] = {}

    def net_at(d, t):
        saved = torch.load(path_of(d, t), map_location=device, weights_only=False)
        key = tuple(sorted(saved["config"].items()))
        if key not in nets:
            nets[key] = StepNet(**saved["config"]).to(device)
            nets[key].bf16, nets[key].fp16 = amp == "bf16", amp == "fp16"
        net = nets[key]
        net.load_state_dict(saved["state_dict"])
        return net, saved

    began = time.time()
    for d in range(K):
        net, saved = net_at(d, 0)
        replayed = dict(groups, train_all=full) if full is not None else groups
        if starts[d] is not None:
            start = starts[d].to(device)
        else:
            assert full is not None
            start = orient(net, full, top_direction(full.R), step_train[(d, 0)], args.node_batch)  # (its sign)
            if args.save_starts:
                saved["start"] = start.cpu()
                saved["state_dict"] = {k: v.cpu() for k, v in saved["state_dict"].items()}
                torch.save(saved, path_of(d, 0) + ".tmp")
                os.replace(path_of(d, 0) + ".tmp", path_of(d, 0))  # (a fresh file: hard links elsewhere unchanged)
        hyper = {s: (start.expand(g.nodes, -1), g.R.new_zeros(g.nodes)) for s, g in replayed.items()}
        ans = {}
        for t in range(steps[d]):
            net = net_at(d, t)[0] if t > 0 else net
            ans = {s: answers(net, g, *hyper[s], args.node_batch) for s, g in replayed.items()}
            if t + 1 < steps[d]:
                hyper = {s: next_hyperplane(*ans[s]) for s in replayed}
        for s, g in replayed.items():
            side = None
            if s == "sample":  # (each side drawn from its node's share: rare with probability m / (1 + m))
                m = ans[s][1].clamp_min(0)[g.node]
                side = torch.rand(len(g.R), device=device, generator=gen) < m / (1 + m)
            descend(g, *hyper[s], *ans[s], 0.75 * torch.cuda.get_device_properties(device).total_memory
                    if device.type == "cuda" else 0.0, bits, side)
        mse["train"].append(mse_now(full) if full is not None else math.nan)
        mse["val"].append(mse_now(groups["val"]))
        mse["test"].append(groups["test"].R[:args.score].square().mean().item())
        if d == last_missing:  # (no start left to recompute: the training images are done)
            full = None
            torch.cuda.empty_cache() if device.type == "cuda" else None
        if d + 1 in lengths:
            for s, g in groups.items():
                k = count if s == "sample" else args.n
                recon[s].append(inputs[s][:k] - g.R[:k])
        r = rec["depths"][d]
        if d % 16 == 0 or d + 1 == K:
            print(f"depth {d}: val mse {mse['val'][-1]:.6f} (run {r['val']:.6f}), test {mse['test'][-1]:.6f} over "
                  f"{min(args.score, len(inputs['test']))} (run {r['test']:.6f} over all)" +
                  (f", train {mse['train'][-1]:.6f} (run {r['train']:.6f})" if not math.isnan(mse['train'][-1])
                   else "") + f"; {time.time() - began:.0f}s", flush=True)

    with open(os.path.join(out, "mse.json"), "w") as f:
        json.dump(dict(run=args.run, depths=K, amp=amp, mse={k: [None if math.isnan(v) else v for v in vs]
                                                             for k, vs in mse.items()}, lengths=lengths), f)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = range(1, K + 1)
    ax.plot(xs, [r["train"] for r in rec["depths"][:K]], color="#eb6834", linewidth=1, alpha=0.5,
            label="train, the run's records")
    ax.plot(xs, [r["test"] for r in rec["depths"][:K]], color="#2a78d6", linewidth=1, alpha=0.5,
            label="test, the run's records")
    ax.plot(xs, [r["val"] for r in rec["depths"][:K]], color="#3aa57a", linewidth=1, alpha=0.5,
            label="val, the run's records")
    if missing:
        ax.plot(xs, mse["train"], color="#eb6834", linewidth=2, linestyle="--", label=f"train, replayed ({amp})")
    ax.plot(xs, mse["val"], color="#3aa57a", linewidth=2, linestyle="--", label=f"val, replayed ({amp})")
    ax.plot(xs, mse["test"], color="#2a78d6", linewidth=2, linestyle="--",
            label=f"test ({min(args.score, len(inputs['test']))} images), replayed")
    ax.set_yscale("log")
    ax.set_xlabel("tokens")
    ax.set_ylabel("mse per pixel")
    ax.set_title(f"{os.path.basename(args.run.rstrip('/'))}: reconstruction error by code length", loc="left")
    ax.grid(True, color="#e2e2df")
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "mse.png"), dpi=150, facecolor="white")

    for s in ("train", "val", "test"):
        grid = torch.stack([x[:args.n] for x in recon[s]], 1)  # [n, 1 + lengths, D]
        save_grid(grid.clamp(0, 1).reshape(-1, *shape), os.path.join(out, f"compress_{s}.png"), nrow=grid.shape[1])
        print(f"compress_{s}.png: rows are {s} images, then their reconstructions after {lengths} tokens")
    samples = torch.stack(recon["sample"], 1)  # [count, lengths, D]
    save_grid(samples[:args.n].clamp(0, 1).reshape(-1, *shape), os.path.join(out, "samples_by_depth.png"),
              nrow=samples.shape[1])
    save_grid(samples[:args.grid ** 2, -1].clamp(0, 1).reshape(-1, *shape), os.path.join(out, "samples.png"),
              nrow=args.grid)
    real = X["test"][torch.randperm(len(X["test"]), generator=torch.Generator().manual_seed(args.seed))[:args.grid ** 2]
                     .to(device)]
    save_grid(real.clamp(0, 1).reshape(-1, *shape), os.path.join(out, "real.png"), nrow=args.grid)
    print(f"samples.png: {args.grid ** 2} codes sampled with the encoder's shares, after {K} tokens; "
          f"samples_by_depth.png: {args.n} of them after {lengths}; real.png: test images; {out}")


if __name__ == "__main__":
    main()
