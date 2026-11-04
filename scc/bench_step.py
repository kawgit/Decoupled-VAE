"""Time train_walk.py's training step with every depth active, and profile where the GPU's time goes.

Run from the repo: python scc/bench_step.py [--max-nodes 256] [--amp] [--benchmark] [--profile]
"""

import argparse
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from dvae.artifacts import artifact  # noqa: E402
from dvae.data import get_device, load_mnist  # noqa: E402
from dvae.deltatree import D, SharedDecoder, pixels, top_direction  # noqa: E402
from dvae.graphs import GraphedDepths  # noqa: E402
from train_walk import refresh, train_step  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depths", type=int, default=32)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lloyd-steps", type=int, default=4)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--max-nodes", type=int, default=256)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--amp", action="store_true", help="bfloat16 autocast")
    p.add_argument("--benchmark", action="store_true", help="cudnn.benchmark")
    p.add_argument("--tf32", action="store_true", help="TF32 matmuls")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--label", default="")
    p.add_argument("--device", default=None)
    p.add_argument("--ckpt", default=None, help="a train_walk.py checkpoint's weights and starts (realistic splits)")
    p.add_argument("--graphs", action="store_true", help="CUDA graphs (dvae/graphs.py)")
    p.add_argument("--check", action="store_true", help="with --graphs, first compare a step with the eager one")
    a = p.parse_args()
    torch.backends.cudnn.benchmark = a.benchmark
    if a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    device = get_device(a.device)

    def report(text):  # to bench.txt in the run's artifacts folder, and the screen
        print(text, flush=True)
        with open(artifact("bench.txt"), "a") as f:
            f.write(text + "\n")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    X = pixels(load_mnist("train")).to(device)
    mean = X.mean(0)
    torch.manual_seed(0)
    R = X - mean
    top = top_direction(R)
    if a.ckpt:  # its own architecture, grown to --depths if shallower
        shared = torch.load(a.ckpt, map_location=device, weights_only=False)["shared"]
        dec = SharedDecoder(**shared["config"]).to(device)
        dec.load_state_dict(shared["state_dict"])
        if dec.depths < a.depths:
            dec.grow(a.depths, top.expand(a.depths - dec.depths, D))
    else:
        dec = SharedDecoder(a.depths, a.width, a.layers, a.heads).to(device)
        with torch.no_grad():
            dec.starts.copy_(top.expand(a.depths, D))
    tracked = dec.starts.detach().clone()
    grams = (R.T @ R / len(R)).expand(dec.depths, D, D).clone()
    opt = torch.optim.AdamW(dec.parameters(), lr=1e-4)
    args = SimpleNamespace(depths=a.depths, max_nodes=a.max_nodes, lloyd_steps=a.lloyd_steps, tol=0.0,
                           dir_weight=1.0, batch=a.batch, ema=0.95)
    gen = torch.Generator().manual_seed(0)
    graphs = GraphedDepths(dec, a.batch, a.batch * D) if a.graphs else None
    caps = [a.lloyd_steps] * dec.depths
    if graphs is not None and a.check:
        for trial in range(3):  # the first records the graphs; the later ones replay them
            idx = torch.randint(len(X), (a.batch,), generator=gen).to(device)
            results = []
            for g in (None, graphs):
                for q in dec.parameters():
                    q.grad = torch.zeros_like(q) if q.grad is None else q.grad.zero_()
                stats, keys = train_step(dec, X[idx], mean, None, args, caps, g)
                results.append((stats, keys, torch.cat([q.grad.flatten() for q in dec.parameters()
                                                        if q.grad is not None])))
            (s0, k0, g0), (s1, k1, g1) = results
            report(f"check {trial}: objective rel diff {((s0[0] - s1[0]).abs() / s0[0].clamp_min(1e-12)).max():.2e}, "
                   f"codes equal {torch.equal(k0, k1)}, grad rel diff {(g0 - g1).norm() / g0.norm():.2e}")

    def step():
        idx = torch.randint(len(X), (a.batch,), generator=gen).to(device)
        opt.zero_grad(set_to_none=graphs is None)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=a.amp):
            stats, _ = train_step(dec, X[idx], mean, grams, args, caps, graphs)
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        return stats

    for _ in range(a.warmup):
        step()
    sync()
    t0 = time.time()
    for _ in range(a.steps):
        stats = step()
    sync()
    per = (time.time() - t0) / a.steps
    t1 = time.time()
    refresh(dec, grams, tracked)
    sync()
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    gpu = torch.cuda.get_device_name() if device.type == "cuda" else device.type
    line = (f"{a.label} on {gpu}: {per:.3f}s per step ({per * 54:.0f}s per epoch of 54 steps); refresh "
            f"{time.time() - t1:.2f}s; peak memory {peak:.1f} GB; nodes per depth {stats[3].round().int().tolist()}")
    report(f"{vars(a)}\n  {line}")
    if a.profile:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            step()
            sync()
        report(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))


if __name__ == "__main__":
    main()
