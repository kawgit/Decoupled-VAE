"""Stage 1: train the outer encoder and decoder, a convolutional autoencoder from images to a small continuous latent
(C channels at 8 x 8 for CIFAR-10's 32 x 32), as compact as it can be while its reconstructions stay good. The inner
encoder (train_inner_components.py --outer) then turns that latent into a discrete code a prior can model.

It trains one autoencoder per latent size in --channels, smallest first, each on pixel mse with early stopping on the
val images' mse (logged every epoch), until every target in --target-mse is met: each target's choice is the first
(most compact) size whose val mse meets it. Every size trained is kept (outer_c<C>.pt), so another target can pick
another later. The latent is --downsample times smaller than the image each way (C x 8 x 8 for 32 x 32 at 4,
C x 16 x 16 for 128 x 128 at 8).

Each target's choice's latents of every train, val and test image go to latents_<target>.pt (and the first target's
to latents.pt too), standardized per channel and scaled to --latent-std (by default about a pixel's spread, so the
inner encoder's settings carry over), with the mean, std and scale that undo it. outer.png: each size's val mse by epoch, and the final val mse against the latent's size, with
the target; pairs_c<C>.png: test images (top) and their reconstructions (bottom).

Run from the repo: python train_outer.py --dataset cifar10 --tf32 --bf16   (V100s: --fp16 in place of --bf16)
Next: train_inner_components.py --outer artifacts/<job> (or --outer artifacts/<job>/latents_<target>.pt).
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
    p.add_argument("--target-mse", default="1e-4", help="val pixel mse targets (comma-separated); each gets the most "
                                                        "compact size that meets it, and its own latents file")
    p.add_argument("--downsample", type=int, default=4, choices=(2, 4, 8, 16), help="image side / latent side")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=2, help="epochs without a better val mse before stopping")
    p.add_argument("--lr-patience", type=int, default=1, help="epochs without a better val mse before halving lr")
    p.add_argument("--latent-std", type=float, default=0.25, help="each latent channel's std in the latents files")
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
    downs = args.downsample.bit_length() - 1
    side = (shape[1] // args.downsample, shape[2] // args.downsample)
    targets = sorted((float(t) for t in args.target_mse.split(",")), reverse=True)  # (loosest first)
    # (uint8 on the host, pinned; each batch goes to the GPU as it is used: 100k 128 x 128 images as floats are 20 GB)
    X = {s: load_images(args.dataset, s, args.data_root)[:lim].view(-1, *shape)
         for s, lim in (("train", args.limit_train), ("val", args.limit_val), ("test", args.limit_test))}
    if device.type == "cuda":
        X = {s: x.pin_memory() for s, x in X.items()}
    eval_batch = max(32, 1_000_000 // (shape[1] * shape[2]))
    # (a fixed random part of the training images, as many as val, scored as val is: the train/val gap each epoch)
    seen = torch.randperm(len(X["train"]), generator=torch.Generator().manual_seed(args.seed + 1))[:len(X["val"])]
    X_seen = X["train"][seen.sort().values]
    if device.type == "cuda":
        X_seen = X_seen.pin_memory()
    log(f"{len(X['train'])} train, {len(X['val'])} val, {len(X['test'])} test images {tuple(shape)}; latents "
        f"C x {side[0]} x {side[1]}; targets {targets}")
    amp = (torch.bfloat16 if args.bf16 else torch.float16) if (args.bf16 or args.fp16) else None

    def autocast():
        return torch.autocast(device.type, dtype=amp, enabled=amp is not None)

    def on_device(x):
        return pixels(x.to(device, non_blocking=True)).view(-1, *shape)

    @torch.no_grad()
    def mse_of(model, x):
        model.eval()
        total = 0.0
        for i in range(0, len(x), eval_batch):
            xb = on_device(x[i:i + eval_batch])
            with autocast():
                y = model(xb)
            total += (y.float() - xb).square().sum().item()
        return total / x.numel()

    @torch.no_grad()
    def encode(model, x):
        model.eval()
        parts = []
        for i in range(0, len(x), eval_batch):
            with autocast():
                parts.append(model.encoder(on_device(x[i:i + eval_batch])).float())
        return torch.cat(parts)

    records: dict = dict(args=vars(args), sizes=[], chosen={})
    chosen: dict[float, int] = {}
    for C in (int(c) for c in args.channels.split(",")):
        model = Outer(C, args.width, shape[0], downs).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=args.lr_patience)
        scaler = torch.amp.GradScaler(device.type, enabled=args.fp16)
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        dims = C * side[0] * side[1]
        best = (float("inf"), 0, copy.deepcopy(model.state_dict()))
        curve, curve_train = [], []
        began = time.time()
        log(f"latent {C} x {side[0]} x {side[1]} = {dims} numbers ({dims / (shape[0] * shape[1] * shape[2]):.3f} of the "
            f"image's); {sum(q.numel() for q in model.parameters()) / 1e6:.2f}M parameters")
        for epoch in range(1, args.max_epochs + 1):
            model.train()
            t0 = time.time()
            for idx in torch.randperm(len(X["train"]), generator=gen).split(args.batch):
                x = on_device(X["train"][idx])
                with autocast():
                    loss = (model(x).float() - x).square().mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            v, tr = mse_of(model, X["val"]), mse_of(model, X_seen)
            sched.step(v)
            curve.append(v)
            curve_train.append(tr)
            if v < best[0]:
                best = (v, epoch, copy.deepcopy(model.state_dict()))
            log(f"  c{C} epoch {epoch}: val mse {v:.6f}, train {tr:.6f} (val {100 * (v / tr - 1):+.1f}%; best val "
                f"{best[0]:.6f} at epoch {best[1]}); lr {opt.param_groups[0]['lr']:.1e}; {time.time() - t0:.0f}s")
            if epoch - best[1] >= args.patience:
                break
        model.load_state_dict(best[2])
        test = mse_of(model, X["test"])
        torch.save(dict(config=model.config, state_dict=model.state_dict(), val=best[0], test=test, epoch=best[1]),
                   os.path.join(out, f"outer_c{C}.pt"))
        met = [t for t in targets if t not in chosen and best[0] <= t]
        for t in met:
            chosen[t] = C
        train_mse = curve_train[best[1] - 1]
        records["sizes"].append(dict(channels=C, dims=dims, val=best[0], test=test, train=train_mse, epochs=len(curve),
                                     best_epoch=best[1], curve=curve, curve_train=curve_train,
                                     seconds=time.time() - began))
        records["chosen"] = {f"{t:g}": c for t, c in chosen.items()}
        with open(os.path.join(out, "records.json"), "w") as f:
            json.dump(records, f)
        with torch.no_grad(), autocast():
            shown = model(on_device(X["test"][:args.pairs])).float().clamp(0, 1)
        save_grid(torch.cat([on_device(X["test"][:args.pairs]), shown]), os.path.join(out, f"pairs_c{C}.png"),
                  nrow=args.pairs)
        log(f"c{C} ({dims} numbers): val mse {best[0]:.6f}, test {test:.6f}, train {train_mse:.6f} (val "
            f"{100 * (best[0] / train_mse - 1):+.1f}%; epoch {best[1]} of {len(curve)}); "
            f"{(time.time() - began) / 60:.1f} min" + (f"; meets {', '.join(f'{t:g}' for t in met)}" if met else ""))
        if len(chosen) == len(targets):
            break

    fig, (by_epoch, by_size) = plt.subplots(1, 2, figsize=(14, 5))
    for r in records["sizes"]:
        line, = by_epoch.plot(range(1, len(r["curve"]) + 1), r["curve"], linewidth=1.6,
                              label=f"{r['dims']} numbers (c{r['channels']}): val")
        by_epoch.plot(range(1, len(r["curve_train"]) + 1), r["curve_train"], linewidth=1.2, linestyle="--",
                      color=line.get_color(), label="train")
    for ax in (by_epoch, by_size):
        for t in targets:
            ax.axhline(t, color="#1d1d1f", linestyle=":", linewidth=1.2)
        ax.set_yscale("log")
        ax.grid(True, color="#e2e2df")
        for side_ in ("top", "right"):
            ax.spines[side_].set_visible(False)
    by_size.plot([r["dims"] for r in records["sizes"]], [r["val"] for r in records["sizes"]], "o-", color="#2a78d6",
                 label="val")
    by_size.plot([r["dims"] for r in records["sizes"]], [r["test"] for r in records["sizes"]], "o--", color="#3aa57a",
                 label="test")
    by_size.plot([r["dims"] for r in records["sizes"]], [r["train"] for r in records["sizes"]], "o:", color="#e8743b",
                 label="train (as many images as val)")
    by_size.set_xscale("log")
    by_epoch.set(title="Pixel mse by epoch (val solid, train dashed; dotted: targets)", xlabel="epoch", ylabel="mse")
    by_size.set(title="Best val pixel mse by latent size", xlabel="numbers per image", ylabel="mse")
    for ax in (by_epoch, by_size):
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "outer.png"), dpi=150, facecolor="white")

    first = float(args.target_mse.split(",")[0])
    for t in (t for t in targets if t not in chosen):
        log(f"no size met the target {t:g}: no latents for it (add larger --channels)")
    for t, C in chosen.items():
        saved = torch.load(os.path.join(out, f"outer_c{C}.pt"), map_location=device, weights_only=False)
        model = Outer(**saved["config"]).to(device)
        model.load_state_dict(saved["state_dict"])
        Z = {s: encode(model, x) for s, x in X.items()}
        mu, sd = Z["train"].mean((0, 2, 3)), Z["train"].std((0, 2, 3))  # (per channel, from the training images)
        scale = args.latent_std
        lat = {s: ((z - mu[None, :, None, None]) / sd[None, :, None, None] * scale).half().cpu() for s, z in Z.items()}
        names = [f"latents_{t:g}.pt"] + (["latents.pt"] if t == first else [])
        for name in names:
            torch.save(dict(**lat, channels=C, shape=tuple(lat["train"].shape[1:]), mean=mu.cpu(), std=sd.cpu(),
                            scale=scale, target=t, dataset=args.dataset,
                            limits=dict(train=args.limit_train, val=args.limit_val, test=args.limit_test)),
                       os.path.join(out, name))
        log(f"target {t:g}: c{C}, latents {tuple(lat['train'].shape)} -> {', '.join(names)}")
    log(f"done; {out}")


if __name__ == "__main__":
    main()
