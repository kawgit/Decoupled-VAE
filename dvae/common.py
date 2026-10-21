"""Device selection, checkpoints, CLI helpers and a small early-stopping training loop."""

import argparse
import itertools
import json
import math
import os
import signal
import sys
import time
from typing import Any

import torch

from .models import MODELS as _MODELS
from .models import MODES
from .staged import StagedEncoder

MODELS = {**_MODELS, "encoder": StagedEncoder}


# Cap what the MPS allocator may take (relative to macOS's recommended GPU working set, ~13 GB on an
# 18 GB machine). PyTorch's default of 1.7 lets it grow far past physical memory, so the system swaps
# and freezes instead of raising an out-of-memory error. Must be set before the first MPS allocation.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")  # must not exceed the high ratio


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
    parser.add_argument("--out", default=out, help="best checkpoint; the latest one goes next to it as *_latest.pt")
    parser.add_argument("--resume", default=None,
                        help="checkpoint to continue from (best or latest); its saved args are reused unless overridden")
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=epochs, help="max epochs")
    parser.add_argument("--patience", type=int, default=patience, help="early-stop after this many epochs without val improvement")
    parser.add_argument("--log-interval", type=float, default=0.5,
                        help="print each step's loss, at most once per this many seconds (0.5 = 2 Hz)")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None, help="for smoke tests")


def parse_train_args(parser):
    """Parse args; with --resume, start from the checkpoint's saved args and apply only the flags given now."""
    args = parser.parse_args()
    if not args.resume:
        return args
    saved = torch.load(args.resume, map_location="cpu", weights_only=False)["args"]
    given = {a.split("=")[0] for a in sys.argv[1:] if a.startswith("-")}
    explicit = {a.dest for a in parser._actions if given & set(a.option_strings)}
    overrides = sorted(explicit - {"resume"})
    print(f"resuming from {args.resume}" + (f", overriding {overrides}" if overrides else ""))
    return argparse.Namespace(**{**saved, **{k: getattr(args, k) for k in explicit}})


def latest_path(out):
    root, ext = os.path.splitext(out)
    return f"{root}_latest{ext}"


def save_checkpoint(path, kind, model, **extra):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"kind": kind, "config": model.config, "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path, kind, device):
    if not os.path.exists(path) and os.path.exists(latest_path(path)):
        print(f"{path} doesn't exist yet (no epoch finished); using {latest_path(path)}")
        path = latest_path(path)
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


class _GracefulInterrupt:
    """First Ctrl-C asks the loop to stop at the next step boundary; a second one aborts immediately."""

    def __enter__(self):
        self.requested = False
        self._old = signal.signal(signal.SIGINT, self._handle)
        return self

    def _handle(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print("\nCtrl-C: saving a checkpoint after this step (Ctrl-C again to abort without saving)", flush=True)

    def __exit__(self, *exc):
        signal.signal(signal.SIGINT, self._old)


def _rng_states():
    states: dict[str, Any] = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        states["mps"] = torch.mps.get_rng_state()
    return states


def _set_rng_states(states):
    torch.set_rng_state(states["torch"])
    if "cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])
    if "mps" in states and torch.backends.mps.is_available():
        torch.mps.set_rng_state(states["mps"])


def fit(model, kind, args, epoch_batches, step_loss, evaluate, gen, **extra):
    """Generic loop: AdamW + linear warmup, early stopping on evaluate()["loss"].

    epoch_batches(epoch) yields the epoch's batches, drawing its shuffle from gen; step_loss(batch)
    returns the scalar loss to minimize, or (loss, logs) where logs is a dict of floats/lists averaged
    over the epoch into metrics["train_logs"]; evaluate() returns a dict of metrics with a "loss" key.

    Saves args.out whenever val improves and *_latest.pt after every epoch and on Ctrl-C. Both hold
    the full training state, so either can be passed to --resume, including mid-epoch.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = lambda step: min(1.0, (step + 1) / max(1, args.warmup_steps))  # noqa: E731
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup)
    # epoch / steps_in_epoch: where to pick up. gen_state: the shuffle generator at the start of that epoch.
    # epoch_losses: this epoch's step losses so far, so a mid-epoch resume still reports the full epoch average.
    state: dict[str, Any] = dict(epoch=1, steps_in_epoch=0, step=0, best=math.inf, bad_epochs=0, history=[],
                                 gen_state=gen.get_state(), epoch_losses=[], epoch_logs=[])
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ckpt["kind"] != kind or "train_state" not in ckpt:
            raise ValueError(f"{args.resume} is not a resumable {kind} checkpoint")
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        # The restored optimizer/scheduler carry the old lr; apply the current --lr / --weight-decay instead.
        sched.base_lrs = [args.lr] * len(opt.param_groups)
        for group in opt.param_groups:
            group.update(initial_lr=args.lr, lr=args.lr * warmup(sched.last_epoch), weight_decay=args.weight_decay)
        state = ckpt["train_state"]
        _set_rng_states(ckpt["rng"])
        print(f"resumed at epoch {state['epoch']} step {state['step']} (best val loss {state['best']:.5f})")
    latest = latest_path(args.out)

    def save(path, metrics):
        save_checkpoint(path, kind, model, args=vars(args), metrics=metrics, history=state["history"],
                        optimizer=opt.state_dict(), scheduler=sched.state_dict(), train_state=state, rng=_rng_states(),
                        **extra)

    with _GracefulInterrupt() as interrupt:
        while state["epoch"] <= args.epochs:
            epoch = state["epoch"]
            gen.set_state(state["gen_state"])
            model.train()
            t0, steps_this_run, last_log = time.time(), 0, -math.inf
            running = state["epoch_losses"]
            for batch in itertools.islice(epoch_batches(epoch), state["steps_in_epoch"], args.max_steps_per_epoch):
                loss = step_loss(batch)
                if isinstance(loss, tuple):
                    loss, logs = loss
                    state.setdefault("epoch_logs", []).append(logs)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                sched.step()
                state["step"] += 1
                state["steps_in_epoch"] += 1
                running.append(loss.item())
                steps_this_run += 1
                now = time.time()
                if now - last_log >= args.log_interval:
                    last_log = now
                    rate = steps_this_run / (now - t0)
                    last_logs = state["epoch_logs"][-1] if state.get("epoch_logs") else {}
                    diag_text = "".join(f", {key} {last_logs[key]:.3f}" for key in ("code_entropy_bits", "distinct_frac")
                                        if key in last_logs)
                    print(f"epoch {epoch} step {state['step']} loss {running[-1]:.5f} "
                          f"(epoch avg {sum(running) / len(running):.5f}, lr {sched.get_last_lr()[0]:.2e}{diag_text}, "
                          f"{rate:.1f} it/s)", flush=True)
                if interrupt.requested:
                    save(latest, None)
                    print(f"saved {latest} at epoch {epoch} step {state['step']}; continue with --resume {latest}")
                    return state["best"], state["history"]

            model.eval()
            with torch.no_grad():
                metrics = evaluate()
            metrics.update(epoch=epoch, train_loss=sum(running) / max(1, len(running)), seconds=time.time() - t0)
            if state.get("epoch_logs"):
                metrics["train_logs"] = _average_logs(state["epoch_logs"])
            state["history"].append(metrics)
            improved = metrics["loss"] < state["best"]
            if improved:
                state["best"], state["bad_epochs"] = metrics["loss"], 0
            else:
                state["bad_epochs"] += 1
            state.update(epoch=epoch + 1, steps_in_epoch=0, gen_state=gen.get_state(), epoch_losses=[], epoch_logs=[])
            print(f"== epoch {epoch} val {json.dumps(_round(metrics))}{'  *saved*' if improved else ''}", flush=True)
            if improved:
                save(args.out, metrics)
            save(latest, metrics)
            if interrupt.requested:
                print(f"saved {latest} after epoch {epoch}; continue with --resume {latest}")
                break
            if state["bad_epochs"] >= args.patience:
                print(f"early stopping after {epoch} epochs (best val loss {state['best']:.5f})")
                break
    print(f"best {kind} (val loss {state['best']:.5f}) is at {args.out}")
    return state["best"], state["history"]


def _average_logs(logs):
    """Average a list of {name: float | list[float]} dicts key by key (lists elementwise)."""
    out = {}
    for key in logs[0]:
        values = [entry[key] for entry in logs]
        if isinstance(values[0], list):
            out[key] = [sum(col) / len(col) for col in zip(*values)]
        else:
            out[key] = sum(values) / len(values)
    return out


def _round(obj):
    if isinstance(obj, float):
        return round(obj, 5)
    if isinstance(obj, list):
        return [_round(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    return obj
