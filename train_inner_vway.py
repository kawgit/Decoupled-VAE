"""Stage 2 with V-way splits: train_inner_components.py's frozen components, one network per (depth, Lloyd step),
with every node split into V children (dvae/vway.py) instead of two. Each token then carries log2(V) bits.

As there: the tree grows a depth at a time and each depth a Lloyd step at a time; step 0 asks about the depth's
start (here plus and minus the top V / 2 principal directions of the training remainders, scaled by their rms),
step t about step t - 1's answers; each network trains on its fixed nodes and is then frozen; epochs per network,
Lloyd steps per depth and depths are each chosen as the n maximizing gain * (1 - discount) ** n - threshold * n,
giving up once n passes the best by a patience; a depth tries at most as many steps as the most any of the last
--lloyd-lookback depths chose (one more on a --lloyd-explore share). A network starts from whichever of the most
recently trained network and the previous depth's at the same step has the lower val mse. Not here: --init and
--resume, state snapshots.

Each depth logs the share of its nodes' cells (train, nodes of at least 2 V images) that hold no image under the
centroids the chosen split was asked about, and under those its predicted means would make; --attraction pulls every predicted mean toward all of its
node's images, so an empty cell's centroid still moves.

What the run keeps (artifacts/<job>/): records.json (the plot's, as train_inner_components.py's, plus each depth's
empty cells), run.png, tree/<split>/dDDD.pt (each image's token, its next node, each next node's parent, child and
every node's shares; with --store-centroids each node's predicted means too) and with --save-models every network
(models/dDDD_sSS.pt; step 0's also holds the depth's start). The project's disk is tight: neither is on by default.

Run from the repo: python train_inner_vway.py --V 4 --outer artifacts/<outer job> --tf32 --bf16
"""

import argparse
import copy
import json
import math
import os
import random
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from dvae.artifacts import artifact  # noqa: E402
from dvae.components import ImageStream, Utilization, best_count, epoch_batches, mse_now, root_group  # noqa: E402
from dvae.data import SHAPES, get_device, load_images  # noqa: E402
from dvae.deltatree import pixels  # noqa: E402
from dvae.logs import RunLog  # noqa: E402
from dvae.outer import decode_latents, load_outer  # noqa: E402
from dvae.vway import (CellSums, VStepNet, attraction_sse, cell_stats, cells_of, pca_start, vanswers, vdescend,  # noqa: E402
                       vsplit_mse, vsplit_sse)


def atomic_save(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--V", type=int, default=4, help="children per node (even)")
    p.add_argument("--attraction", type=float, default=0.0,
                   help="also pull every predicted mean toward all of its node's images, this much")
    p.add_argument("--depth-threshold", type=float, default=0.0)
    p.add_argument("--lloyd-threshold", type=float, default=0.0)
    p.add_argument("--epoch-threshold", type=float, default=0.0)
    p.add_argument("--lloyd-discount", type=float, default=0.05)
    p.add_argument("--epoch-discount", type=float, default=0.005)
    p.add_argument("--depth-patience", type=int, default=20)
    p.add_argument("--lloyd-patience", type=int, default=2)
    p.add_argument("--lloyd-lookback", type=int, default=100)
    p.add_argument("--lloyd-explore", type=float, default=0.2)
    p.add_argument("--epoch-patience", type=int, default=2)
    p.add_argument("--max-depths", type=int, default=100000)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--images-per-epoch", type=int, default=22528)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--head", choices=("token", "shared"), default="token",
                   help="the predicted means' image head: one call per centroid token, or one per node (V channels)")
    p.add_argument("--head-hidden", type=int, default=1, help="multiplies the image head's convolutions' channels")
    p.add_argument("--side-m-history", action="store_true", help="history tokens also read each step's token and share")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--max-nodes", type=int, default=0,
                   help="nodes per forward pass while training (0: a batch's, in one pass; fewer passes launch fewer "
                        "small kernels, which bound the loop: 256 took 2.1 times as long per epoch on ImageNet "
                        "latents)")
    p.add_argument("--node-batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--history-window", type=int, default=16)
    p.add_argument("--table-bits", type=int, choices=(8, 16), default=8)
    p.add_argument("--gpu-tables", type=float, default=0.75)
    p.add_argument("--centroid-dtype", choices=("float32", "bfloat16"), default="float32",
                   help="the dtype of every node's V centroids (the steps' queries and answers, [nodes, V, D] each): "
                        "bfloat16 halves them, for large D (the split codes the rounded centroids)")
    p.add_argument("--save-models", action="store_true", help="keep every network in models/")
    p.add_argument("--store-centroids", action="store_true", help="keep every node's predicted means in tree/")
    p.add_argument("--plot-pairs", type=int, default=12)
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true", help="float16 autocast with a gradient scaler (V100s)")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the network's memory and query (fewer, fused kernels: the training loop is "
                        "otherwise bound by the CPU launching small ones)")
    p.add_argument("--profile-depth", type=int, default=None,
                   help="profile this depth's first step's training (torch.profiler: profile.txt, the kernels by GPU "
                        "time) and stop")
    p.add_argument("--hours", type=float, default=0.0)
    p.add_argument("--outer", default=None, metavar="RUN", help="a train_outer.py run: code its latents")
    p.add_argument("--px-images", type=int, default=2000)
    p.add_argument("--px-every", type=int, default=10,
                   help="decode the --px-images images for the pixel mse every this many depths (and at the first)")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--limit-test", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="data")
    p.add_argument("--dataset", choices=sorted(SHAPES), default="cifar10")
    args = p.parse_args()
    V = args.V
    out = args.out or os.path.dirname(artifact("run.log"))
    for sub in ("models", "tree/train", "tree/val", "tree/test"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    device = get_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    log = RunLog(os.path.join(out, "run.log"), device)
    util = Utilization(device)
    log(f"=== train_inner_vway.py {time.strftime('%Y-%m-%d')} {vars(args)}")
    began = time.time()

    limits = (("train", args.limit_train), ("val", args.limit_val), ("test", args.limit_test))
    outer, lat = load_outer(args.outer, device) if args.outer else (None, None)
    if lat is not None:
        args.dataset = lat["dataset"]
        data = {s: lat[s][:lim].float().flatten(1).to(device) for s, lim in limits}
    else:
        data = {s: pixels(load_images(args.dataset, s, args.data_root)[:lim]).to(device) for s, lim in limits}
    mean = data["train"].mean(0)
    groups = {s: root_group(s, X, mean, args.history_window) for s, X in data.items()}
    train, val, test = groups["train"], groups["val"], groups["test"]
    N, D = train.R.shape
    max_nodes = args.max_nodes or args.batch
    total_memory = torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 0
    budget = args.gpu_tables * total_memory
    shape = tuple(lat["shape"]) if lat is not None else SHAPES[args.dataset]
    cdtype = getattr(torch, args.centroid_dtype)
    torch.manual_seed(args.seed)

    def amp():
        return torch.autocast(device.type, dtype=torch.bfloat16 if args.bf16 else torch.float16,
                              enabled=args.bf16 or args.fp16)

    images_px = {s: pixels(load_images(args.dataset, s, args.data_root)[:min(len(data[s]), args.px_images)]).to(device)
                 for s in ("train", "val", "test")} if lat is not None else {}

    # (the raw data past what pixel_mse and test_pairs read: the groups hold the remainders)
    data = {s: X[:max(args.px_images, args.plot_pairs)].clone() for s, X in data.items()}

    def as_pixels(s, R):
        if outer is None or lat is None:
            return data[s][:len(R)] - R
        return decode_latents(outer, lat, data[s][:len(R)] - R, amp, batch=250)  # (128 x 128 images: 1000 is 4 GB)

    def pixel_mse():
        return {f"{s}_px": (as_pixels(s, groups[s].R[:len(x)]) - x).square().mean().item() for s, x in images_px.items()}

    net = VStepNet(V, args.width, args.layers, args.heads, not args.side_m_history, shape, args.dropout,
                   args.head, args.head_hidden).to(device)
    net.bf16, net.fp16 = args.bf16, args.fp16
    if args.compile:  # (dynamic: the batch's nodes and the history's length vary)
        # (each function compiles a few times at the first depths, for grad mode, batches of one node and the
        # history's alignment; past the limit it would run eagerly without saying so)
        torch._dynamo.config.recompile_limit = 64
        setattr(net, "memory", torch.compile(net.memory, dynamic=True))
        setattr(net, "query", torch.compile(net.query, dynamic=True))
    log(f"{N} train, {len(val.R)} val, {len(test.R)} test images of {D} numbers; V = {V} ({math.log2(V):.2f} bits a "
        f"token); each network {sum(q.numel() for q in net.parameters()) / 1e6:.2f}M parameters")
    root_val = mse_now(val)
    records: dict = dict(args=vars(args), mean=mean.cpu().tolist(), models=[], depths=[], kept=0,
                         root=dict(train=mse_now(train), val=root_val, test=mse_now(test), **pixel_mse()))
    drawing: list[subprocess.Popen] = []

    def plot(final=False):
        if drawing and drawing[0].poll() is None and not final:
            return
        if drawing:
            if drawing[0].wait() != 0:
                log(f"plot failed (exit {drawing[0].returncode})")
            drawing.clear()
        drawing.append(subprocess.Popen([sys.executable, "-m", "dvae.plots", os.path.join(out, "records.json"),
                                         os.path.join(out, "run.png")], cwd=os.path.dirname(os.path.abspath(__file__))))
        if final:
            drawing[0].wait()

    recent: dict | None = None
    prev_depth: dict[int, dict] = {}
    chosen_steps: list[int] = []
    depth_gains: list[float] = []
    gen = torch.Generator().manual_seed(args.seed)
    stream = ImageStream(N, gen)
    epoch_steps = max(1, round(args.images_per_epoch / args.batch))

    def val_mse(model, st_va: tuple[torch.Tensor, CellSums, torch.Tensor], q_va):
        return vsplit_mse(val, st_va[0], st_va[1], vanswers(model, val, q_va, args.node_batch)[0])

    def train_net(model, st_tr, st_va, q_tr, q_va, v0, timing, label):
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup_steps))
        best = (0.0, 0, v0, copy.deepcopy(model.state_dict()))
        scaler = torch.amp.GradScaler(device.type, enabled=args.fp16)
        count, total, _ = st_tr
        assert train.stats is not None
        e = 0
        while True:
            t0 = time.time()
            model.train()
            for pieces, B in epoch_batches(train, stream.take(epoch_steps * args.batch), args.batch, max_nodes):
                opt.zero_grad(set_to_none=True)
                for nodes, w in pieces:
                    c, _ = model.query(model.memory(*train.history(nodes)), q_tr[nodes])
                    sse = vsplit_sse(c, count[nodes], total[nodes])
                    if args.attraction > 0:
                        sse = sse + args.attraction * attraction_sse(c, train.stats.count[nodes], train.stats.total[nodes])
                    scaler.scale((sse * w).sum() / (B * D)).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
            e += 1
            t1 = time.time()
            timing["train"] += t1 - t0
            v = val_mse(model, st_va, q_va)
            objective = (v0 - v) * (1 - args.epoch_discount) ** e - args.epoch_threshold * e
            if objective > best[0]:
                best = (objective, e, v, copy.deepcopy(model.state_dict()))
            log.detail(f"      {label} epoch {e}: val mse {v:.6f} (gain {v0 - v:.2e}, objective {objective:.2e}; "
                       f"best epoch {best[1]})")
            timing["val"] += time.time() - t1
            if e - best[1] >= args.epoch_patience or e >= args.max_epochs:
                break
        model.load_state_dict(best[3])
        return e, best[1]

    def empty_share(count):
        """The share of empty cells among the nodes holding at least 2 V images (a node with fewer has some
        anyway): centroids that win nothing."""
        big = count.sum(-1) >= 2 * V
        return (count[big] == 0).float().mean().item() if big.any() else 0.0

    parts: dict[str, float] = {}  # (what a depth's "other" seconds went to, logged with it)

    def part(name, t0):
        """Add the seconds since t0 (after the GPU's queued work) to parts[name]; returns now."""
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.time()
        parts[name] = parts.get(name, 0.0) + t1 - t0
        return t1

    for d in range(args.max_depths):
        depth_began = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        timing_depth = dict(train=0.0, val=0.0, other=0.0)
        parts.clear()
        t_other = time.time()
        start = pca_start(train.R, V)
        tp = part("start", t_other)
        before = dict(train=mse_now(train), val=mse_now(val))
        lineage = ",".join(map(str, chosen_steps))
        query = {s: start.to(cdtype).expand(groups[s].nodes, -1, -1).contiguous() for s in ("train", "val")}
        part("start", tp)
        timing_depth["other"] += time.time() - t_other
        gains: list[float] = []
        states: dict[int, dict] = {}
        best_n = 0
        best_split: dict | None = None
        cap = args.max_steps
        if args.lloyd_lookback > 0 and len(chosen_steps) >= args.lloyd_lookback:
            cap = max(chosen_steps[-args.lloyd_lookback:])
            if random.Random(args.seed * 1_000_003 + d).random() < args.lloyd_explore:
                cap += 1
            cap = min(cap, args.max_steps)
        for t in range(cap):
            timing = dict(train=0.0, val=0.0, other=0.0)
            t_other = time.time()
            st_tr, st_va = cell_stats(train, query["train"]), cell_stats(val, query["val"])
            part("cells", t_other)
            candidates = [(label, s) for label, s in (("most recent", recent), (f"d{d - 1}s{t}", prev_depth.get(t)))
                          if s is not None]
            timing["other"] += time.time() - t_other
            t_val = time.time()
            if not candidates:
                start_label, v0 = "fresh", val_mse(net, st_va, query["val"])
            else:
                scored = []
                for label, state in candidates:
                    net.load_state_dict(state)
                    scored.append((val_mse(net, st_va, query["val"]), label, state))
                v0, start_label, state = min(scored, key=lambda x: x[0])
                net.load_state_dict(state)
            timing["val"] += time.time() - t_val
            if d == args.profile_depth and t == 0:
                from torch.profiler import ProfilerActivity, profile
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    train_net(net, st_tr, st_va, query["train"], query["val"], v0, timing, f"d{d}s{t}")
                table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=60, max_name_column_width=90)
                with open(os.path.join(out, "profile.txt"), "w") as f:
                    f.write(table)
                log(f"profiled d{d}s{t}'s training: {os.path.join(out, 'profile.txt')}")
                return
            epochs, kept_epoch = train_net(net, st_tr, st_va, query["train"], query["val"], v0, timing, f"d{d}s{t}")
            t_other = time.time()
            c_va, p_va = vanswers(net, val, query["val"], args.node_batch)
            tp = part("answers", t_other)
            after_val = vsplit_mse(val, *st_va[:2], c_va)
            after: dict[str, float | None] = dict(val=after_val)
            gains.append(before["val"] - after_val)
            n_best = best_count(gains, args.lloyd_threshold, args.lloyd_discount)
            if n_best == t + 1:
                best_n = n_best
            last = t + 1 - best_n >= args.lloyd_patience or t + 1 >= cap
            # (the train nodes' answers are the split if this step is kept, else the next step's question: a
            # depth's last step, which is rarely kept, needs neither, so its train mse goes unmeasured)
            c_tr = p_tr = None
            after["train"] = None
            if best_n == t + 1:
                best_split = None  # (the step it beat: freed before this one's train answers are made)
            if best_n == t + 1 or not last:
                tp = time.time()
                c_tr, p_tr = vanswers(net, train, query["train"], args.node_batch)
                tp = part("answers", tp)
                after["train"] = vsplit_mse(train, *st_tr[:2], c_tr)
            if best_n == t + 1:
                # (the split's cells rather than the query that made them: the query, [nodes, V, D], can go)
                best_split = dict(train=(st_tr[2], c_tr, p_tr, st_tr[0]), val=(st_va[2], c_va, p_va))
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            states[t] = state
            recent = state
            if args.save_models:
                atomic_save(dict(config=net.config, state_dict={k: v.cpu() for k, v in state.items()}, d=d, t=t,
                                 **(dict(start=start.cpu()) if t == 0 else {})),
                            os.path.join(out, "models", f"d{d:03d}_s{t:02d}.pt"))
            part("step mse", tp)
            timing["other"] += time.time() - t_other
            seconds = sum(timing.values())
            for k in timing:
                timing_depth[k] += timing[k]
            records["models"].append(dict(d=d, t=t, lineage=lineage, start=start_label, epochs=epochs,
                                          kept_epoch=kept_epoch, val_start=v0, val=after["val"], train=after["train"],
                                          gain=gains[-1], empty=empty_share(st_tr[0]), seconds=timing))
            step_gain = gains[-1] - (gains[-2] if t > 0 else 0.0)
            log(f"  d{d} s{t}: {epochs} epochs (kept {kept_epoch}), from {start_label} (val {v0:.6f}); val mse "
                f"{after_val:.6f} (step gain {step_gain:+.2e}, depth gain {gains[-1]:.2e}, best {best_n} steps), "
                f"train {'-' if after['train'] is None else format(after['train'], '.6f')}; empty cells {empty_share(st_tr[0]):.1%}; {seconds:.1f}s: "
                f"{timing['train']:.1f} train, {timing['val']:.1f} val, {timing['other']:.1f} other")
            if last:
                break
            assert c_tr is not None
            query = dict(train=c_tr, val=c_va)
        assert best_split is not None
        empty_best = empty_share(best_split["train"][3])
        # (the steps' [nodes, V, D] tensors that the split doesn't use: on ImageNet each is gigabytes)
        del query, st_tr, st_va, c_tr, c_va, p_tr, p_va
        t_other = time.time()
        q_te = start.to(cdtype).expand(test.nodes, -1, -1).contiguous()
        c_te = p_te = q_te
        for t in range(best_n):
            net.load_state_dict(states[t])
            c_te, p_te = vanswers(net, test, q_te, args.node_batch)
            if t + 1 < best_n:
                q_te = c_te
        tp = part("test answers", t_other)
        splits = dict(train=best_split["train"][:3], val=best_split["val"], test=(cells_of(test, q_te), c_te, p_te))
        nodes, empty_after = {}, empty_share(cell_stats(train, best_split["train"][1])[0])
        tp = part("cells", tp)
        for s, (cells, c, pr) in splits.items():
            g = groups[s]
            nodes[s] = g.nodes
            kept = vdescend(g, cells, c, pr, budget, args.table_bits)
            atomic_save(dict(V=V, c=c.half().cpu() if args.store_centroids else None, start=start.cpu(), **kept),
                        os.path.join(out, "tree", s, f"d{d:03d}.pt"))
        del splits, best_split, c, pr, cells, q_te, c_te, p_te
        tp = part("split", tp)
        chosen_steps.append(best_n)
        prev_depth = states
        now = dict(train=mse_now(train), val=mse_now(val), test=mse_now(test))
        tp = part("depth mse", tp)
        if d % args.px_every == 0:
            now.update(pixel_mse())
        k = args.plot_pairs
        real = images_px["test"][:k] if lat is not None else data["test"][:k]
        atomic_save(dict(real=real.cpu(), recon=as_pixels("test", test.R[:k]).float().cpu(),
                         shape=SHAPES[args.dataset], tokens=d + 1), os.path.join(out, "test_pairs.pt"))
        part("pixels", tp)
        depth_gains.append(root_val - now["val"])
        k_best = best_count(depth_gains, args.depth_threshold)
        timing_depth["other"] += time.time() - t_other
        elapsed = time.time() - began
        busy = util.mean(depth_began)
        peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
        records["depths"].append(dict(d=d, steps=best_n, tried=len(gains), gains=gains, **now,
                                      gain=depth_gains[-1] - (depth_gains[-2] if d > 0 else 0.0), nodes=nodes,
                                      bits=(d + 1) * math.log2(V), empty=empty_best,
                                      empty_after=empty_after, seconds=elapsed, wall=time.time() - depth_began,
                                      timing=timing_depth, other=dict(parts), util=busy,
                                      peak_gib=peak))
        records["kept"] = k_best
        atomic_json(records, os.path.join(out, "records.json"))
        plot()
        wall = time.time() - depth_began
        log(f"depth {d} ({(d + 1) * math.log2(V):.0f} bits): {best_n} of {len(gains)} Lloyd steps; mse val "
            f"{now['val']:.6f} (gain {records['depths'][-1]['gain']:.2e}; best {k_best} depths), train "
            f"{now['train']:.6f}, test {now['test']:.6f}"
            + (f" (pixels: train {now['train_px']:.6f}, val {now['val_px']:.6f}, test {now['test_px']:.6f})"
               if "val_px" in now else "")
            + f"; empty cells {records['depths'][-1]['empty']:.1%} (after: {empty_after:.1%}); nodes "
            f"{nodes['train']} train, {nodes['val']} val; {wall:.0f}s: {timing_depth['train']:.0f} train, "
            f"{timing_depth['val']:.0f} val, {timing_depth['other']:.0f} other ("
            + ", ".join(f"{k} {v:.1f}" for k, v in parts.items()) + ")"
            + (f"; GPU {busy:.0f}% busy" if busy is not None else "")
            + (f", peak {peak:.1f} GiB" if peak is not None else "") + f"; {elapsed / 3600:.2f} h in all")
        if d + 1 - k_best >= args.depth_patience:
            log(f"no depth past {k_best} paid {args.depth_threshold:.1e} for {args.depth_patience} depths: done")
            break
        if args.hours > 0 and time.time() - began > args.hours * 3600:
            log(f"--hours {args.hours} reached")
            break
    plot(final=True)
    best = records["depths"][records["kept"] - 1]
    log(f"the tree keeps {records['kept']} depths ({records['kept'] * math.log2(V):.0f} bits): val mse "
        f"{best['val']:.6f}, test {best['test']:.6f}"
        + (f" (pixels: val {best['val_px']:.6f}, test {best['test_px']:.6f})" if "val_px" in best else "") + f"; {out}")


if __name__ == "__main__":
    main()
