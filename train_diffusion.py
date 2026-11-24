"""Single-GPU unconditional CIFAR-10 DDPM baseline with EMA and resumable atomic checkpoints."""

import argparse
from contextlib import nullcontext
import copy
import json
import math
from pathlib import Path
import platform
import random
import signal
import subprocess
import sys
import time

import torch
from torch.optim.swa_utils import get_ema_multi_avg_fn
from torchvision.utils import save_image

from dvae.artifacts import artifact
from dvae.data import get_device, load_images
from dvae.diffusion import Diffusion, UNet


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=None, help="new run directory; defaults to artifacts/$DVAE_JOB")
    p.add_argument("--resume", help="latest.pt, best.pt or final.pt; restores optimizer, EMA, scaler and RNG")
    p.add_argument("--device", default=None)
    p.add_argument("--data-root", default="data")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--diffusion-steps", type=int, default=1000)
    p.add_argument("--schedule", choices=("linear", "cosine"), default="linear")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=5000)
    p.add_argument("--ema", type=float, default=0.9999)
    p.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    p.add_argument("--max-steps", type=int, default=200000, help="total steps including steps restored by resume")
    p.add_argument("--max-hours", type=float, default=23.0, help="wall-clock cap for this invocation, 0: unlimited")
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--sample-every", type=int, default=10000)
    p.add_argument("--preview-steps", type=int, default=100, help="DDIM steps; final preview is full DDPM")
    p.add_argument("--sample-count", type=int, default=64)
    p.add_argument("--val-images", type=int, default=1024)
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-flip", action="store_true")
    p.add_argument("--limit-train", type=int, default=None, help="smoke tests only; recorded in checkpoint")
    args = p.parse_args()
    restored = None
    if args.resume:
        restored = torch.load(args.resume, map_location="cpu", weights_only=False)
        explicit = {v.split("=")[0] for v in sys.argv[1:] if v.startswith("--")}
        changes = {key: value for key, value in vars(args).items() if "--" + key.replace("_", "-") in explicit}
        fixed = ("width", "blocks", "dropout", "diffusion_steps", "schedule", "ema", "precision", "batch",
                 "lr", "warmup", "seed", "no_flip", "limit_train")
        for key in fixed:
            if key in changes and changes[key] != restored["args"][key]:
                p.error(f"cannot change --{key.replace('_', '-')} when resuming")
        args = argparse.Namespace(**{**restored["args"], **changes})
    if min(args.batch, args.max_steps, args.save_every, args.eval_every, args.sample_every, args.eval_batch,
           args.log_every, args.sample_count, args.val_images, args.width, args.blocks) < 1:
        p.error("batch sizes, widths, counts, steps and intervals must be positive")
    if args.width % 4 or args.diffusion_steps < 2 or not 0 < args.ema < 1 or args.max_hours < 0:
        p.error("width must be divisible by 4; diffusion steps >= 2; EMA in (0,1); hours >= 0")
    if not 1 <= args.preview_steps <= args.diffusion_steps or args.lr <= 0 or args.warmup < 0:
        p.error("invalid preview steps, learning rate or warmup")
    return args, restored


def main():
    args, restored = parse_args()
    began = time.monotonic()
    device = get_device(args.device)
    if args.precision != "fp32" and device.type != "cuda":
        raise ValueError("bf16/fp16 training requires CUDA; use --precision fp32 for CPU smoke tests")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16; choose --precision fp16 or fp32")
    out = Path(args.out or artifact(""))
    if restored is None and out.exists() and any(out.glob("*.pt")):
        raise ValueError(f"{out} already has checkpoints; choose a new --out or use --resume")
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)
    args.out = str(out)

    def log(message):
        line = time.strftime("%H:%M:%S ") + message
        print(line, flush=True)
        with (out / "run.log").open("a") as stream:
            stream.write(line + "\n")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    model = UNet(args.width, blocks=args.blocks, dropout=args.dropout).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    ema_update = get_ema_multi_avg_fn(args.ema)
    diffusion = Diffusion(args.diffusion_steps, args.schedule, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")

    def amp():
        if args.precision == "fp32":
            return nullcontext()
        return torch.autocast("cuda", dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16)

    # Exactly the same seed-0 split as the tree experiments; test images are never loaded here.
    train = load_images("cifar10", "train", args.data_root)[:args.limit_train]
    val = load_images("cifar10", "val", args.data_root)[:args.val_images]
    if not len(train) or not len(val):
        raise ValueError("empty train or validation set")
    if device.type == "cuda":
        train, val = train.pin_memory(), val.pin_memory()
    records = [] if restored is None else restored["history"]
    step, best, gpu_seconds = 0, math.inf, 0.0
    if restored is not None:
        model.load_state_dict(restored["model"])
        ema.load_state_dict(restored["ema"])
        optimizer.load_state_dict(restored["optimizer"])
        scaler.load_state_dict(restored["scaler"])
        step, best = restored["step"], restored["best_val"]
        gpu_seconds = restored["wall_seconds"]
        torch.set_rng_state(restored["torch_rng"])
        random.setstate(restored["python_rng"])
        if device.type == "cuda" and restored["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(restored["cuda_rng"])
    del restored
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--short"], stderr=subprocess.DEVNULL, text=True)
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, "Git metadata unavailable (common for SCC sync copies)"
    parameters = sum(p.numel() for p in model.parameters())
    manifest = dict(args=vars(args), model=model.config, train_images=len(train), val_images=len(val),
                    dataset="cifar10", split_seed=0, validation_selection="first val-images of fixed 5000 split",
                    parameters=parameters, torch=str(torch.__version__),
                    python=platform.python_version(), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
                    commit=commit, git_status=dirty, note="Local DDPM baseline, no published FID claim")
    (out / f"manifest_step{step:07d}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Retain exact implementation even when SCC has no .git directory.
    source_dir = out / f"source_step{step:07d}"
    source_dir.mkdir(exist_ok=True)
    import shutil
    for file in (Path(__file__), Path(__file__).parent / "dvae" / "diffusion.py",
                 Path(__file__).parent / "dvae" / "data.py"):
        shutil.copy2(file, source_dir / file.name)
    log(f"start/resume step {step}: {parameters / 1e6:.2f}M parameters; {len(train)} train, "
        f"{len(val)} val; {manifest['gpu']}; {args.precision}; out {out}")
    stopping = [False]

    def request_stop(signum, frame):
        stopping[0] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def plot():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if records:
            axes[0].plot([r["step"] for r in records], [r["val_epsilon_mse"] for r in records])
        axes[0].set(xlabel="Optimizer step", ylabel="EMA validation epsilon MSE",
                    title="Fixed validation noise; not FID or reconstruction MSE")
        files = sorted((out / "samples").glob("step*.png"))
        if files:
            axes[1].imshow(plt.imread(files[-1]))
            axes[1].set_title(files[-1].name)
        else:
            axes[1].text(0.5, 0.5, "Samples after the first preview interval", ha="center")
        axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(out / "run.png", dpi=140)
        plt.close(fig)

    def save(name):
        atomic_save(dict(format="cifar10-ddpm-v1", args=vars(args), config=model.config, model=model.state_dict(),
                         ema=ema.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                         step=step, best_val=best, history=records, wall_seconds=gpu_seconds + time.monotonic() - began,
                         torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                         cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None), out / name)
        (out / "records.json").write_text(json.dumps(dict(manifest=manifest, step=step, best_val=best,
                                                          history=records), indent=2) + "\n")
        plot()

    @torch.no_grad()
    def evaluate():
        gen = torch.Generator(device=device).manual_seed(12345)
        total = 0.0
        for offset in range(0, len(val), args.eval_batch):
            x = val[offset:offset + args.eval_batch].to(device).float() / 127.5 - 1
            t = torch.randint(diffusion.steps, (len(x),), device=device, generator=gen)
            noise = torch.randn(x.shape, device=device, generator=gen)
            with amp():
                loss = (ema(diffusion.noisy(x, t, noise), t).float() - noise).square().mean((1, 2, 3))
            total += loss.sum().item()
        return total / len(val)

    def preview(final=False):
        gen = torch.Generator(device=device).manual_seed(2026)
        sampler = "ddpm" if final else "ddim"
        batches = [diffusion.sample(ema, min(args.eval_batch, args.sample_count - i), gen, sampler,
                                   None if final else args.preview_steps, amp).cpu()
                   for i in range(0, args.sample_count, args.eval_batch)]
        save_image(torch.cat(batches), out / "samples" / f"step{step:07d}_{sampler}.png", nrow=8)
        plot()
        log(f"saved EMA {sampler} samples at step {step}")

    running, last, last_step = [], time.monotonic(), step
    model.train()
    try:
        while step < args.max_steps and not stopping[0]:
            if args.max_hours and time.monotonic() - began >= args.max_hours * 3600:
                log("wall-clock budget reached; saving for resume")
                break
            idx = torch.randint(len(train), (args.batch,))
            x = train[idx].to(device, non_blocking=True).float() / 127.5 - 1
            if not args.no_flip:
                flip = torch.rand(len(x), device=device) < 0.5
                x = torch.where(flip[:, None, None, None], x.flip(-1), x)
            t = torch.randint(diffusion.steps, (len(x),), device=device)
            noise = torch.randn_like(x)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
            optimizer.zero_grad(set_to_none=True)
            with amp():
                loss = (model(diffusion.noisy(x, t, noise), t).float() - noise).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                log("fp16 overflow: optimizer step skipped")
                continue
            step += 1
            with torch.no_grad():
                ema_update(list(ema.parameters()), list(model.parameters()), step)
            running.append(loss.detach())
            if step % args.log_every == 0:
                elapsed = time.monotonic() - last
                log(f"step {step}/{args.max_steps}: train epsilon MSE {torch.stack(running).mean().item():.6f}; "
                    f"{(step - last_step) / elapsed:.2f} steps/s; invocation {(time.monotonic() - began) / 3600:.2f} h")
                running, last, last_step = [], time.monotonic(), step
            if step % args.eval_every == 0:
                value = evaluate()
                records.append(dict(step=step, val_epsilon_mse=value, hours=(gpu_seconds + time.monotonic() - began) / 3600))
                log(f"step {step}: EMA val epsilon MSE {value:.6f} (not image MSE, FID or bits/dim)")
                if value < best:
                    best = value
                    save("best.pt")
            if step % args.save_every == 0:
                save("latest.pt")
            if step % args.sample_every == 0 and not stopping[0]:
                preview()
    finally:
        save("latest.pt")
        log(f"saved resumable latest.pt at step {step}")
    if not stopping[0]:
        value = evaluate()
        records.append(dict(step=step, val_epsilon_mse=value, hours=(gpu_seconds + time.monotonic() - began) / 3600))
        if value < best:
            best = value
            save("best.pt")
        save("final.pt")
        # The checkpoint is already durable if this slower sampling phase reaches scheduler termination.
        preview(final=True)
    log(f"finished invocation at step {step}; wall seconds {time.monotonic() - began:.1f}")


if __name__ == "__main__":
    main()
