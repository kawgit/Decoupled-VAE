"""Stage 1: train the outer encoder and decoder, a convolutional autoencoder from images to a small continuous latent
(C channels at 8 x 8 for CIFAR-10's 32 x 32), as compact as it can be while its reconstructions stay good. The inner
encoder (train_inner_components.py --outer) then turns that latent into a discrete code a prior can model.

It trains one autoencoder per latent size in --channels, smallest first, each on pixel mse with early stopping on the
val images' mse (logged every epoch), and stops at the first whose val mse meets --target-mse: the most compact one
that is good enough. Every size trained is kept (outer_c<C>.pt), so another target can pick another later.

The chosen one's latents of every train, val and test image go to latents.pt, standardized per channel and scaled to
--latent-std (by default about a pixel's spread, so the inner encoder's settings carry over), with the mean, std and
scale that undo it. outer.png: each size's val mse by epoch, and the final val mse against the latent's size, with
the target; pairs_c<C>.png: test images (top) and their reconstructions (bottom).

Run from the repo: python train_outer.py --dataset cifar10 --tf32 --bf16   (V100s: --fp16 in place of --bf16)
Next: train_inner_components.py --outer artifacts/<job>.
"""

import argparse
import copy
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from dvae.artifacts import artifact  # noqa: E402
from dvae.data import SHAPES, get_device, load_images  # noqa: E402
from dvae.deltatree import pixels  # noqa: E402
from dvae.logs import RunLog  # noqa: E402
from dvae.outer import Outer  # noqa: E402
from dvae.viz import save_grid  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channels", default="4,8,12,16,24,32,48", help="latent channels to try, smallest first")
    p.add_argument("--target-mse", type=float, default=1e-4, help="val pixel mse the chosen size must reach")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=8, help="epochs without a better val mse before stopping")
    p.add_argument("--lr-patience", type=int, default=3, help="epochs without a better val mse before halving lr")
    p.add_argument("--latent-std", type=float, default=0.25, help="each latent channel's std in latents.pt")
    p.add_argument("--pairs", type=int, default=12)
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--limit-test", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="data")
    p.add_argument("--dataset", choices=sorted(SHAPES), default="cifar10")
    args = p.parse_args()
    out = args.out or os.path.dirname(artifact("run.log"))
    os.makedirs(out, exist_ok=True)
    device = get_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = args.tf32
    log = RunLog(os.path.join(out, "run.log"), device)
    log(f"=== train_outer.py {time.strftime('%Y-%m-%d')} {vars(args)}")
    torch.manual_seed(args.seed)
    shape = SHAPES[args.dataset]
    X = {s: pixels(load_images(args.dataset, s, args.data_root)[:lim]).view(-1, *shape).to(device)
         for s, lim in (("train", args.limit_train), ("val", args.limit_val), ("test", args.limit_test))}
    log(f"{len(X['train'])} train, {len(X['val'])} val, {len(X['test'])} test images {tuple(shape)}")
    amp = (torch.bfloat16 if args.bf16 else torch.float16) if (args.bf16 or args.fp16) else None

    def autocast():
        return torch.autocast(device.type, dtype=amp, enabled=amp is not None)

    @torch.no_grad()
    def mse_of(model, x, batch=1000):
        model.eval()
        total = 0.0
        for i in range(0, len(x), batch):
            with autocast():
                y = model(x[i:i + batch])
            total += (y.float() - x[i:i + batch]).square().sum().item()
        return total / x.numel()

    @torch.no_grad()
    def encode(model, x, batch=1000):
        model.eval()
        parts = []
        for i in range(0, len(x), batch):
            with autocast():
                parts.append(model.encoder(x[i:i + batch]).float())
        return torch.cat(parts)

    records: dict = dict(args=vars(args), sizes=[])
    chosen = None
    for C in (int(c) for c in args.channels.split(",")):
        model = Outer(C, args.width, shape[0]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=args.lr_patience)
        scaler = torch.amp.GradScaler(device.type, enabled=args.fp16)
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        dims = C * (shape[1] // 4) * (shape[2] // 4)
        best = (float("inf"), 0, copy.deepcopy(model.state_dict()))
        curve = []
        began = time.time()
        log(f"latent {C} x {shape[1] // 4} x {shape[2] // 4} = {dims} numbers ({dims / (shape[0] * shape[1] * shape[2]):.3f}"
            f" of the image's); {sum(q.numel() for q in model.parameters()) / 1e6:.2f}M parameters")
        for epoch in range(1, args.max_epochs + 1):
            model.train()
            t0 = time.time()
            for idx in torch.randperm(len(X["train"]), generator=gen).split(args.batch):
                x = X["train"][idx.to(device)]
                with autocast():
                    loss = (model(x).float() - x).square().mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            v = mse_of(model, X["val"])
            sched.step(v)
            curve.append(v)
            if v < best[0]:
                best = (v, epoch, copy.deepcopy(model.state_dict()))
            log(f"  c{C} epoch {epoch}: val mse {v:.6f} (best {best[0]:.6f} at epoch {best[1]}); lr "
                f"{opt.param_groups[0]['lr']:.1e}; {time.time() - t0:.0f}s")
            if epoch - best[1] >= args.patience:
                break
        model.load_state_dict(best[2])
        test = mse_of(model, X["test"])
        torch.save(dict(config=model.config, state_dict=model.state_dict(), val=best[0], test=test, epoch=best[1]),
                   os.path.join(out, f"outer_c{C}.pt"))
        records["sizes"].append(dict(channels=C, dims=dims, val=best[0], test=test, epochs=len(curve),
                                     best_epoch=best[1], curve=curve, seconds=time.time() - began))
        with open(os.path.join(out, "records.json"), "w") as f:
            json.dump(records, f)
        with torch.no_grad(), autocast():
            shown = model(X["test"][:args.pairs]).float().clamp(0, 1)
        save_grid(torch.cat([X["test"][:args.pairs], shown]), os.path.join(out, f"pairs_c{C}.png"), nrow=args.pairs)
        log(f"c{C} ({dims} numbers): val mse {best[0]:.6f}, test {test:.6f} (epoch {best[1]} of {len(curve)}); "
            f"{(time.time() - began) / 60:.1f} min; target {args.target_mse:.1e} "
            + ("met" if best[0] <= args.target_mse else "not met"))
        if best[0] <= args.target_mse:
            chosen = (C, model)
            break

    fig, (by_epoch, by_size) = plt.subplots(1, 2, figsize=(14, 5))
    for r in records["sizes"]:
        by_epoch.plot(range(1, len(r["curve"]) + 1), r["curve"], linewidth=1.6, label=f"{r['dims']} numbers (c{r['channels']})")
    for ax in (by_epoch, by_size):
        ax.axhline(args.target_mse, color="#1d1d1f", linestyle=":", linewidth=1.2, label="target")
        ax.set_yscale("log")
        ax.grid(True, color="#e2e2df")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    by_size.plot([r["dims"] for r in records["sizes"]], [r["val"] for r in records["sizes"]], "o-", color="#2a78d6",
                 label="val")
    by_size.plot([r["dims"] for r in records["sizes"]], [r["test"] for r in records["sizes"]], "o--", color="#3aa57a",
                 label="test")
    by_size.set_xscale("log")
    by_epoch.set(title="Val pixel mse by epoch", xlabel="epoch", ylabel="mse")
    by_size.set(title="Best val pixel mse by latent size", xlabel="numbers per image", ylabel="mse")
    for ax in (by_epoch, by_size):
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "outer.png"), dpi=150, facecolor="white")

    if chosen is None:
        log(f"no size met the target {args.target_mse:.1e}: no latents.pt (add larger --channels)")
        return
    C, model = chosen
    Z = {s: encode(model, x) for s, x in X.items()}
    mu, sd = Z["train"].mean((0, 2, 3)), Z["train"].std((0, 2, 3))  # (per channel, from the training images)
    scale = args.latent_std
    lat = {s: ((z - mu[None, :, None, None]) / sd[None, :, None, None] * scale).half().cpu() for s, z in Z.items()}
    torch.save(dict(**lat, channels=C, shape=tuple(lat["train"].shape[1:]), mean=mu.cpu(), std=sd.cpu(), scale=scale,
                    dataset=args.dataset, limits=dict(train=args.limit_train, val=args.limit_val, test=args.limit_test)),
               os.path.join(out, "latents.pt"))
    log(f"chose c{C}: latents {tuple(lat['train'].shape)} -> latents.pt; {out}")


if __name__ == "__main__":
    main()
