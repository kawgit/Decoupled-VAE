"""Device selection, checkpoints, CLI helpers and a small early-stopping training loop."""

import json
import math
import os
import time

import torch

from .models import MODELS, MODES


def get_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def add_common_args(parser):
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default="data")


def add_mode_args(parser, who, default="actual"):
    parser.add_argument(f"--{who}-mode", choices=MODES, default=default,
                        help=f"how {who} tokens are drawn: actual distribution, greedy argmax, or top-p nucleus")
    parser.add_argument(f"--{who}-top-p", type=float, default=0.9)


def add_train_args(parser, out, batch_size, lr, epochs, patience):
    parser.add_argument("--out", default=out)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=epochs, help="max epochs")
    parser.add_argument("--patience", type=int, default=patience, help="early-stop after this many epochs without val improvement")
    parser.add_argument("--log-interval", type=float, default=0.5,
                        help="print each step's loss, at most once per this many seconds (0.5 = 2 Hz)")


def save_checkpoint(path, kind, model, **extra):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"kind": kind, "config": model.config, "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path, kind, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt["kind"] != kind:
        raise ValueError(f"{path} is a {ckpt['kind']} checkpoint, expected {kind}")
    model = MODELS[kind](**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def check_same_codes(a, b, what):
    for key in ("vocab_size", "k_max"):
        if a[key] != b[key]:
            raise ValueError(f"{what}: {key} mismatch ({a[key]} vs {b[key]})")


def fit(model, args, train_epoch, evaluate, save):
    """Generic loop: AdamW + linear warmup, early stopping on evaluate()["loss"].

    train_epoch(epoch) yields scalar losses (one per step) after they've been computed with grad;
    evaluate() returns a dict of metrics with a "loss" key; save(metrics, history) writes the best
    checkpoint.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0, (step + 1) / max(1, args.warmup_steps)))
    best, bad_epochs, history, step = math.inf, 0, [], 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, last_log = time.time(), [], -math.inf
        for loss in train_epoch(epoch):
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            step += 1
            running.append(loss.item())
            now = time.time()
            if now - last_log >= args.log_interval:
                last_log = now
                rate = len(running) / (now - t0)
                print(f"epoch {epoch} step {step} loss {running[-1]:.5f} (epoch avg {sum(running) / len(running):.5f}, "
                      f"{rate:.1f} it/s)", flush=True)
        model.eval()
        with torch.no_grad():
            metrics = evaluate()
        metrics.update(epoch=epoch, train_loss=sum(running) / max(1, len(running)), seconds=time.time() - t0)
        history.append(metrics)
        improved = metrics["loss"] < best
        print(f"== epoch {epoch} val {json.dumps(_round(metrics))}{'  *saved*' if improved else ''}", flush=True)
        if improved:
            best, bad_epochs = metrics["loss"], 0
            save(metrics, history)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stopping after {epoch} epochs (best val loss {best:.5f})")
                break
    return best, history


def _round(obj):
    if isinstance(obj, float):
        return round(obj, 5)
    if isinstance(obj, list):
        return [_round(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    return obj
