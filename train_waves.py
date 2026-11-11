"""Train the encoder (train_encoder.py's shared network) in waves, depth by depth.

A wave is the time between two halvings of the threshold (in val mse per GPU second, as train_encoder.py's
--step-threshold). Each wave sweeps the tree from the root: one depth at a time is the focus, trained alone
(each batch starting at it from the prefix cache, dvae/persist.py's PrefixCache, and backpropagated there
only) until its val mse stagnates; then one pass routes every training image through it to refresh its row
of the cache, its val mse then becomes its target for the rest of the wave, and
the focus moves one depth down. The depths the sweep has passed are not improved but maintained: a fixed
monitor set of held-out images is routed down the tree as the network now splits it, and a depth whose val
mse drifts more than --band above its target gets maintenance batches (trained there alone, from the cache)
until it is back, the shallowest first, at most every other batch. So the network's behaviour at those depths
may change while their val mse does not, and the cached paths stay valid if no longer current: the next wave
refreshes them.

Past the deepest depth the sweep grows the tree while a new depth's expected first step gain beats the
threshold times its cost; when it does not, the threshold halves and a new wave starts at the root. A focus
depth's Lloyd steps are chosen as train_encoder.py's threshold schedule chooses them (the most cumulative val
gain less the threshold's price per step, here a step's GPU seconds per pass over the training images,
timed), and stay fixed once its stage ends.

The monitor costs at most about --monitor-share of the time: a check comes once training since the last has
taken (1 - share) / share times as long as the last check. Every --val-every seconds the whole val set is
routed down the whole tree, logged in train_encoder.py's format and saved (model.pt, state.pt, grams.bin and
the cache, which persists across jobs, so --resume carries on mid-wave). --resume also takes a
train_encoder.py checkpoint: the first wave then starts at the root under its threshold.

Run from the repo: python train_waves.py --dataset cifar10 --threshold 0.05 --tf32
"""

import argparse
import itertools
import math
import os
import time

import torch
from torch import Tensor

from dvae.artifacts import artifact
from dvae.data import SHAPES, get_device, load_images
from dvae.deltatree import Encoder, children, choose, lloyd, pixels, root_frontier, top_direction
from dvae.logs import RunLog
from dvae.persist import PrefixCache, load_checkpoint, save_checkpoint
from dvae.plots import plot_waves
from train_encoder import DepthClock, depth_step, evaluate, fold, grow, lloyd_path, psnr, refresh


def explore(n):
    """The steps a focus depth that uses n tries, to see whether more would pay: half again as many."""
    return max(n + 1, math.ceil(1.5 * n))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depths", type=int, default=0, help="the deepest the tree grows; 0: without end")
    p.add_argument("--resume", default=None, metavar="CKPT",
                   help="a checkpoint of this script's or train_encoder.py's (that starts a wave at the root)")
    p.add_argument("--threshold", type=float, default=0.05,
                   help="the first wave's threshold, val mse per GPU second (a resume keeps its own)")
    p.add_argument("--lloyd-steps", type=int, default=8, help="a new depth's first Lloyd steps")
    p.add_argument("--monitor", type=int, default=1000, help="held-out images the checks route")
    p.add_argument("--monitor-share", type=float, default=0.1, help="the share of time the checks may take")
    p.add_argument("--min-batches", type=int, default=8, help="batches between checks at least")
    p.add_argument("--stagnation", type=float, default=1e-3,
                   help="a focus depth stagnates once its val mse falls by less than this share of its best...")
    p.add_argument("--patience", type=int, default=3, help="...over this many checks in a row")
    p.add_argument("--max-stage-passes", type=float, default=30.0,
                   help="a focus stage ends after this many passes over the training images, stagnant or not")
    p.add_argument("--band", type=float, default=5e-3,
                   help="a passed depth gets maintenance once its val mse is this share above its target")
    p.add_argument("--maintain", choices=("objective", "distill"), default="objective",
                   help="a maintenance batch's loss at a passed depth: the training objective on its images' split, "
                        "or distill, the squared error of each image's predicted delta (on its cached side) against "
                        "its cached one, the depth's answers when its stage ended; distill backpropagates only "
                        "the last Lloyd step")
    p.add_argument("--explore-chance", type=float, default=0.5,
                   help="a focus batch's chance of running explore(n) Lloyd steps instead of its n")
    p.add_argument("--gain-smoothing", type=float, default=0.8,
                   help="the weight a running average of per-step gains and costs keeps each check")
    p.add_argument("--val-every", type=float, default=900.0, help="seconds between full val passes and saves")
    p.add_argument("--hours", type=float, default=0.0, help="stop after this long; 0: never")
    p.add_argument("--refresh-images", type=int, default=2048, help="images per piece of a cache refresh")
    p.add_argument("--cache-dir", default=None, help="the prefix cache's folder (default: cache/ beside --out)")
    p.add_argument("--side-m-history", action="store_true", help="as train_encoder.py's")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--batch", type=int, default=1024, help="training images per step")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--grad-steps", type=int, default=8, help="as train_encoder.py's")
    p.add_argument("--grad-tilt", type=float, default=0.1, help="as train_encoder.py's")
    p.add_argument("--max-nodes", type=int, default=256, help="as train_encoder.py's")
    p.add_argument("--node-batch", type=int, default=512, help="nodes split at once when routing without a graph")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--ema", type=float, default=0.95, help="running Gram matrices' decay per step")
    p.add_argument("--refresh", type=int, default=50, help="steps between start updates")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--out", default=None, help="checkpoint (default: model.pt in the run's artifacts folder)")
    p.add_argument("--log", default=None, help="summary log (default: run.log in the run's artifacts folder)")
    p.add_argument("--plot", default=None, help="progress plot (default: --log's path as .png)")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="data")
    p.add_argument("--dataset", choices=sorted(SHAPES), default="mnist")
    args = p.parse_args()
    args.out, args.log = args.out or artifact("model.pt"), args.log or artifact("run.log")
    args.tol, args.dir_weight = 0.0, 1.0  # (depth_step's; no tolerance, no seeds)

    device = get_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    log = RunLog(args.log, device)
    ckpt = load_checkpoint(args.resume, device, log) if args.resume else None
    if ckpt is not None:
        cfg = ckpt["shared"]["config"]
        assert not cfg["seeded"] and not cfg.get("node_mean", False), "waves keep no seeds or node means"
        args.width, args.layers, args.heads = cfg["width"], cfg["layers"], cfg["heads"]
        args.side_m_history = not cfg.get("plain_history", False)
        args.dataset = next(name for name, known in SHAPES.items() if known == tuple(cfg.get("shape", (1, 28, 28))))
    log(f"=== {time.strftime('%Y-%m-%d')} train_waves.py {vars(args)}")
    X = pixels(load_images(args.dataset, "train", args.data_root)[:args.limit_train]).to(device)
    V = pixels(load_images(args.dataset, "val", args.data_root)[:args.limit_val]).to(device)
    M = V[:args.monitor]
    N, D = X.shape
    K = args.depths if args.depths > 0 else math.inf
    mean = X.mean(0) if ckpt is None else ckpt["mean"].to(device)
    root_top, root_gram = top_direction(X - mean), (X - mean).T @ (X - mean) / N
    power = 20 if D > 1024 else 0

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    if ckpt is None:
        encoder = Encoder(1, args.width, args.layers, args.heads, plain_history=not args.side_m_history,
                          shape=SHAPES[args.dataset]).to(device)
        with torch.no_grad():
            encoder.starts[0] = root_top
    else:
        encoder = Encoder(**ckpt["shared"]["config"]).to(device)
        encoder.load_state_dict(ckpt["shared"]["state_dict"])
    encoder.bf16 = args.bf16
    tracked = torch.cat([encoder.starts.detach().clone(), root_top[None]])[:min(encoder.depths + 1, K)]
    grams = root_gram.expand(len(tracked), D, D).clone()
    if ckpt is not None and ckpt.get("grams") is not None:
        kept = ckpt["grams"].to(device)[:len(tracked)]
        grams[:len(kept)] = kept
        tracked[:len(kept)] = ckpt["tracked"].to(device)[:len(kept)]
    opt = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    caps = [args.lloyd_steps] * encoder.depths
    smoothed: list[list[float]] = [[] for _ in caps]  # each depth's running per-step val gains
    step_cost: list[float] = [math.nan] * len(caps)  # GPU seconds per pass over the training images: a step
    over_cost: list[float] = [math.nan] * len(caps)  # and the depth's overhead
    tau, wave, focus = args.threshold, 0, 0
    targets: dict[int, float] = {}  # each passed depth's val mse to keep this wave
    history, step, number = [], 0, 0
    seconds = dict(train=0.0, monitor=0.0, refresh=0.0, val=0.0)
    base = 0.0  # (a branched run's clock goes on from the run it branched from)
    stages: list[dict] = []  # each depth's stage ending: seconds, wave, depth, its target
    if ckpt is not None:
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
            for group in opt.param_groups:
                group.update(lr=args.lr, initial_lr=args.lr, weight_decay=args.weight_decay)
        history, step, number = ckpt["metrics"], ckpt.get("step", 0), ckpt["epoch"] + 1
        caps = list(ckpt.get("caps") or caps)
        smoothed = (list(ckpt.get("smoothed") or []) + [[] for _ in caps])[:len(caps)]
        state = ckpt.get("waves")
        if state is not None:  # (this script's: carry on mid-wave)
            tau, wave, focus = state["threshold"], state["wave"], state["focus"]
            targets = {int(k): v for k, v in state["targets"].items()}
            step_cost, over_cost, seconds = list(state["step_cost"]), list(state["over_cost"]), dict(state["seconds"])
            base, stages = state.get("base"), list(state.get("stages", []))
            if base is None:  # (from before the clock went on from a branched run's: put its entries on it)
                base = next((h.get("seconds", 0.0) for h in reversed(history) if "wave" not in h), 0.0)
                history = [dict(h, seconds=h["seconds"] + base) if "wave" in h else h for h in history]
        else:
            tau = ckpt.get("threshold", tau)
            base = history[-1].get("seconds", 0.0) if history else 0.0
        log(f"resumed {args.resume} after its epoch {ckpt['epoch']} (step {step}): wave {wave} at depth {focus}, "
            f"threshold {tau:.2e}, {encoder.depths} depths")
    step0 = step
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (step0 + s + 1) / args.warmup_steps))
    folder = args.cache_dir or os.path.join(os.path.dirname(args.out), "cache")
    cache = PrefixCache(N, math.inf, D, folder, log)
    if ckpt is None or ckpt.get("waves") is None:
        cache.valid.zero_()  # (a new sweep: rows are refreshed from the root down)
    reach = int(cache.valid.min())
    if focus > reach:  # (a job killed after the checkpoint went on to refresh rows of a newer wave: the paths
        # cached to reach are still valid, so the sweep goes on from there, the targets above it kept)
        log(f"the cache reaches depth {reach} for every image, not {focus}: the sweep goes on from {reach}")
        focus, targets = reach, {j: v for j, v in targets.items() if j < reach}
    clock = DepthClock(device.type == "cuda")
    log(f"shared encoder: {sum(q.numel() for q in encoder.parameters()) / 1e6:.1f}M parameters, "
        f"{encoder.depths} depths; {N} training images, {len(V)} val ({len(M)} monitored)")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def batches():
        while True:
            yield from torch.randperm(N, generator=gen).split(args.batch)

    stream = batches()

    def frontier_at(idx, d):
        Xb = X[idx.to(device)]
        return root_frontier(Xb, mean, resident=True) if d == 0 else cache.frontier(idx, Xb, mean, d)

    def train_batch(d, n):
        """One batch trained at depth d alone, from the cache, with n Lloyd steps: its objective (summed)
        and nodes."""
        nonlocal step
        idx = next(stream)
        fr = frontier_at(idx, d)
        fold(grams, d, fr.R, args.ema)
        opt.zero_grad(set_to_none=True)
        began = clock.start()
        sse, _, s = depth_step(encoder, fr, args, n, 1.0, clock)
        clock.stop(d, began)
        if d == encoder.depths - 1 and len(grams) > encoder.depths:  # (track the next depth's start)
            routed = clock.start()
            child, _ = children(fr, s)
            clock.stop(d, routed, "route")
            fold(grams, d + 1, child.R, args.ema)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % args.refresh == 0:
            refresh(encoder, grams, tracked, power)
        return sse / (len(idx) * D), len(s.steps)

    def distill_batch(d, n):
        """One maintenance batch at depth d by --maintain distill, with n Lloyd steps: each image's delta, as the
        network now answers at its node on its cached side, regressed on its cached delta (the network's answer
        when the stage ended), the gradient through the last step only. Returns the squared error per pixel."""
        nonlocal step
        idx = next(stream)
        fr = frontier_at(idx, d)
        target = cache.deltas[d][idx].to(device).float()
        rare = cache.sides[d][idx].to(device) > 0
        fold(grams, d, fr.R, args.ema)
        opt.zero_grad(set_to_none=True)
        began = clock.start()
        total = fr.R.new_zeros(())
        for first in range(0, len(fr.bits), args.max_nodes):
            nodes = torch.arange(first, min(first + args.max_nodes, len(fr.bits)), device=device)
            mine = (fr.node >= first) & (fr.node < first + len(nodes))
            node = fr.node[mine] - first
            mems = encoder.memory(*fr.history(nodes, device))
            r = m = None
            for _, _, _, r, m, _ in lloyd(encoder, mems, fr.starts(encoder, nodes), n, 0.0, {n - 1}):
                pass
            assert r is not None and m is not None
            r, m = r.float(), m.float()
            pred = torch.where(rare[mine, None], r[node], -m[node, None] * r[node])
            sse = (pred - target[mine]).square().sum()
            (sse / (args.batch * D)).backward()
            total = total + sse.detach()
        clock.stop(d, began)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % args.refresh == 0:
            refresh(encoder, grams, tracked, power)
        return total / (len(idx) * D)

    @torch.no_grad()
    def monitor(d, probe):
        """The monitor set's val mse after each depth to d [d + 1], routed as the network now splits it, and
        depth d's per-step gains for probe steps (routing with its caps[d])."""
        encoder.eval()
        fr, mses, gains = root_frontier(M, mean, resident=True), [], []
        for j in range(d + 1):
            s, errs = lloyd_path(encoder, fr, probe if j == d else caps[j], args.node_batch, caps[j])
            if j == d:
                before = torch.cat([fr.R.square().sum(-1)[None], errs[:-1]])
                gains = ((before - errs).mean(1) / D).tolist()
            fr, _ = children(fr, s)
            mses.append(fr.R.square().mean().item())
        encoder.train()
        return mses, gains

    @torch.no_grad()
    def refresh_row(d):
        """Every training image's split at depth d into the cache (a node's split depends only on its history,
        not its images, so any piece of them gives the same one; pieces in key order hold fewer nodes)."""
        encoder.eval()
        order = torch.argsort(cache.keys[d], stable=True) if d > 0 else torch.arange(N)
        for idx in order.split(args.refresh_images):
            fr = frontier_at(idx, d)
            child, _ = children(fr, choose(encoder, fr, caps[d], args.node_batch))
            at = child.node.to(child.deltas.device)
            cache.write_row(idx, d, child.deltas[at, -1], child.sides[at, -1], child.ms[at, -1], child.bits[at])
        encoder.train()

    def add_depth():
        nonlocal tracked, grams
        depth = encoder.depths
        refresh(encoder, grams, tracked, power)
        grow(encoder, opt, depth + 1, tracked[depth:depth + 1])
        caps.append(args.lloyd_steps)
        smoothed.append([])
        step_cost.append(math.nan)
        over_cost.append(math.nan)
        if depth + 1 < K:
            tracked = torch.cat([tracked, root_top[None]])
            grams = torch.cat([grams, root_gram[None]])

    def choose_cap(d):
        """Depth d's Lloyd steps: the most cumulative val gain less the threshold's price per step."""
        if not smoothed[d] or math.isnan(step_cost[d]):
            return caps[d]
        total = list(itertools.accumulate(smoothed[d]))
        return max(range(1, len(total) + 1), key=lambda n: total[n - 1] - tau * step_cost[d] * n)

    def save():
        save_checkpoint(args.out, dict(
            args=vars(args), mean=mean.cpu(), epoch=number, step=step, metrics=history, best=(math.inf, -1),
            shared=dict(config=encoder.config, state_dict={k: v.cpu() for k, v in encoder.state_dict().items()}),
            caps=caps, optimizer=opt.state_dict(), tracked=tracked.cpu(), smoothed=smoothed, threshold=tau,
            waves=dict(threshold=tau, wave=wave, focus=focus, targets=targets, step_cost=step_cost,
                       over_cost=over_cost, seconds=seconds, base=base, stages=stages)), grams)
        cache.flush()

    def full_val(note):
        nonlocal number
        sync()
        t = time.time()
        val, val_steps, _, _ = evaluate(encoder, V, mean, args, caps)
        encoder.train()
        seconds["val"] += time.time() - t
        history.append(dict(epoch=number, val_mse=val, caps=list(caps), threshold=tau, wave=wave, focus=focus,
                            seconds=base + sum(seconds.values()), train_seconds=seconds["train"]))
        plot_waves(history, stages, args.plot or os.path.splitext(args.log)[0] + ".png")
        log(f"  epoch {number:2d}: val mse after {encoder.depths} tokens {val[-1]:.6f} (psnr {psnr(val[-1]):.2f} "
            f"dB); per depth {' '.join(f'{v:.5f}' for v in val)}; wave {wave} at depth {focus} ({note}), threshold "
            f"{tau:.2e}; lloyd steps per depth {' '.join(map(str, caps))}; seconds "
            + ", ".join(f"{k} {v:.0f}" for k, v in seconds.items()))
        save()
        number += 1

    began = time.time()
    last_val = time.time()
    while args.hours <= 0 or time.time() - began < args.hours * 3600:
        # one focus stage at depth focus
        d = focus
        stage_began, passes, best, since, checks = time.time(), 0.0, math.inf, 0, 0
        violated: list[int] = []
        maintained: dict[int, int] = {}
        last_monitor = 0.0
        # the stage's wall seconds by use (training split between the focus depth and maintenance by their shares
        # of its timed GPU work), and the GPU seconds of each and the focus batches
        spent = dict(learning=0.0, maintenance=0.0, checks=0.0, refresh=0.0, val=0.0)
        busy = dict(learning=0.0, maintenance=0.0)
        learned = 0
        log.detail(f"    wave {wave} (threshold {tau:.2e}): depth {d} in focus, {caps[d]} Lloyd steps")
        while True:
            sync()
            t, batch_count, focus_nodes, focus_batches = time.time(), 0, 0, 0
            focus_sse = torch.zeros((), device=device)
            distilled = torch.zeros((), device=device)  # (--maintain distill: its error, summed over batches)
            kept = 0  # maintenance batches this interval
            runs = backs = 0
            while batch_count < args.min_batches or (time.time() - t) < last_monitor * (1 - args.monitor_share) \
                    / args.monitor_share:
                if violated and batch_count % 2 == 1:  # a maintenance batch, the violated depths in turn
                    j = violated[(batch_count // 2) % len(violated)]
                    if args.maintain == "distill":
                        distilled = distilled + distill_batch(j, caps[j])
                    else:
                        train_batch(j, caps[j])
                    maintained[j] = maintained.get(j, 0) + 1
                    kept += 1
                else:
                    n = explore(caps[d]) if float(torch.rand(1, generator=gen)) < args.explore_chance else caps[d]
                    sse, nodes = train_batch(d, n)
                    focus_sse, focus_nodes, focus_batches = focus_sse + sse, focus_nodes + nodes, focus_batches + 1
                    runs, backs = runs + n, backs + (n if args.grad_steps <= 0 else min(n, args.grad_steps))
                batch_count += 1
            parts = clock.totals(encoder.depths)
            sync()
            wall = time.time() - t
            seconds["train"] += wall
            passes += focus_batches * args.batch / N
            learned += focus_batches
            work = [a + r for a, r in zip(parts["depth"], parts["route"])]
            mine, theirs = work[d], sum(work) - work[d]
            share = mine / (mine + theirs) if mine + theirs > 0 else 1.0
            spent["learning"] += wall * share
            spent["maintenance"] += wall * (1 - share)
            busy["learning"] += mine
            busy["maintenance"] += theirs
            # depth d's costs per pass over the training images, from its timed parts (as train_encoder.py's)
            per_pass = N / args.batch
            forward = parts["steps"][d] / max(runs, 1)
            back = max(parts["backward"][d] - 2 * parts["memory"][d], 0.0) / max(backs, 1)
            over = max(parts["depth"][d] + parts["route"][d] - forward * runs - back * backs, 0.0) / max(focus_batches, 1)
            b = args.gain_smoothing
            new_step, new_over = (forward + back) * per_pass, over * per_pass
            step_cost[d] = new_step if math.isnan(step_cost[d]) else b * step_cost[d] + (1 - b) * new_step
            over_cost[d] = new_over if math.isnan(over_cost[d]) else b * over_cost[d] + (1 - b) * new_over
            # the check
            t = time.time()
            mses, gains = monitor(d, explore(caps[d]))
            last_monitor = time.time() - t
            seconds["monitor"] += last_monitor
            cache.touch()  # (a stage writes the cache only at its end)
            spent["checks"] += last_monitor
            old = smoothed[d]
            smoothed[d] = [b * a + (1 - b) * x for a, x in zip(old, gains)] + gains[len(old):] + old[len(gains):]
            caps[d] = choose_cap(d)
            checks += 1
            if mses[d] < best * (1 - args.stagnation):
                best, since = mses[d], 0
            else:
                since += 1
            violated = [j for j in range(d) if mses[j] > targets[j] * (1 + args.band)]
            log.detail(f"    check {checks} at depth {d}: val mse {mses[d]:.6f} (best {best:.6f}, {since} without "
                       f"gain), train objective {focus_sse.item() / max(focus_batches, 1):.6f}, {focus_nodes / max(focus_batches, 1):.0f} "
                       f"nodes per batch, {passes:.2f} passes; steps {caps[d]} (gains "
                       + " ".join(f"{g:.1e}" for g in smoothed[d]) + f"; a step {step_cost[d]:.2f} s per pass, "
                       f"overhead {over_cost[d]:.2f} s); maintenance so far {maintained}"
                       + (f" (distill error {distilled.item() / kept:.2e} per pixel)" if kept and args.maintain == "distill"
                          else "") + "; drifting "
                       + (" ".join(f"{j}:{mses[j] / targets[j] - 1:+.2%}" for j in violated) or "none")
                       + f"; check {last_monitor:.1f}s")
            if time.time() - last_val >= args.val_every:
                t = time.time()
                full_val("mid-stage")
                last_val = time.time()
                spent["val"] += last_val - t
            if since >= args.patience or passes >= args.max_stage_passes:
                break
        targets[d] = mses[d]
        stages.append(dict(seconds=base + sum(seconds.values()), wave=wave, depth=d, target=mses[d]))
        sync()
        t = time.time()
        refresh_row(d)
        sync()
        spent["refresh"] = time.time() - t
        seconds["refresh"] += spent["refresh"]
        trained = spent["learning"] + spent["maintenance"]
        log(f"    wave {wave}: depth {d} done after {checks} checks, {passes:.1f} passes: val mse {mses[d]:.6f} (its "
            f"target), {caps[d]} Lloyd steps; {time.time() - stage_began:.0f}s: learning {spent['learning']:.0f}s "
            f"({learned} batches), maintenance {spent['maintenance']:.0f}s ({sum(maintained.values())} batches"
            + (", by depth " + " ".join(f"{j}:{c}" for j, c in sorted(maintained.items())) if maintained else "")
            + f"), checks {spent['checks']:.0f}s, cache refresh {spent['refresh']:.0f}s"
            + (f", full val {spent['val']:.0f}s" if spent["val"] else "")
            + f"; GPU busy {100 * (busy['learning'] + busy['maintenance']) / max(trained, 1e-9):.0f}% of training")
        if d + 1 < encoder.depths:
            focus = d + 1
            continue
        # past the deepest depth: grow while a new depth's first step pays, else the next wave
        nodes_now = max(focus_nodes / max(focus_batches, 1), 1.0)
        ratio = min(2 * nodes_now, float(args.batch)) / nodes_now
        gain = smoothed[d][0] if smoothed[d] else math.inf
        price = tau * (step_cost[d] + over_cost[d]) * ratio
        if encoder.depths < K and gain > price:
            add_depth()
            focus = d + 1
            log(f"    grow to {encoder.depths} depths: the deepest's first step gains {gain:.2e} > {price:.2e}")
        else:
            log(f"    wave {wave} ends at {encoder.depths} depths (the deepest's first step gains {gain:.2e}, a new "
                f"depth costs {price:.2e}): threshold halves to {tau / 2:.2e}")
            full_val("wave end")
            last_val = time.time()
            tau, wave, focus, targets = tau / 2, wave + 1, 0, {}
    full_val("time is up")
    log(f"stopped after {args.hours} hours: {encoder.depths} depths, saved {args.out}")


if __name__ == "__main__":
    main()
