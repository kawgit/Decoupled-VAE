"""Stage 1: train the encoder as frozen components, one network per (depth, Lloyd step) (dvae/components.py).

The tree grows one depth at a time and each depth one Lloyd step at a time. Step 0 of a depth asks about the
depth's start (the top principal direction of the training remainders there, through each node's predicted
mean); step t about the bisector of step t - 1's answers. Each step's network trains on its fixed nodes (every
network above it frozen) and is then frozen itself. Three counts are each chosen the same way, as the n
maximizing (val mse gain after n) - threshold * n, giving up once n passes the best by a patience:

- epochs, per network (--epoch-threshold, --epoch-patience): the gain is the val mse after the step's split,
  under the network at its start less under it after n epochs; the network kept is the best epoch's. An epoch is
  --images-per-epoch training images (one val measurement after each), taken from one endless stream of shuffles
  of the training set that every network reads in turn, so it means the same however big the dataset is;
- Lloyd steps, per depth (--lloyd-threshold, --lloyd-patience): the val mse before the depth less after n steps'
  split;
- depths (--depth-threshold, --depth-patience): the val mse at the root less after n depths.

So each threshold is in val mse per pixel per epoch, per step and per depth. A depth is a token while steps and
epochs cost only time, so unless given the step threshold is the depth threshold / 25 and the epoch threshold the
step threshold / 40. Steps and depths tried past the best
(within the patience) are trained and kept on disk but are not part of the tree. A network starts from whichever
of these has the lowest val mse at its step: the most recently trained network, the previous depth's network at
the same step, and with --init the earlier run's network at the same (depth, step).

Speed: a node's history is its last --history-window (8) deltas plus one token for the sum of those before, so a
network's cost and the history tables kept stop growing with depth; the tables are 8-bit (--table-bits) and stay on
the GPU. On CIFAR-10 an L40S runs 120 depths in ~0.3 h at ~95% GPU (deep depths ~8 s each, ~65% training, ~12% val).
V100s have no bfloat16: leave --bf16 off there (--fp16 uses their float16 tensor cores instead).

--init RUN reruns from another run (e.g. with lower thresholds): while every choice so far is RUN's (the same
steps at every depth above, the same networks), the step's data is RUN's, and RUN's network is reused as it is,
without training; from the first difference on each network trains, starting from RUN's among the candidates.
--resume is --init from this run's own folder, so an interrupted run replays what it has and goes on.

What the run keeps in its folder (artifacts/<job>/ by default), which the prior and decoder read instead of a
dump: models/dDDD_sSS.pt (every network trained), tree/<split>/dDDD.pt (for train, val and test, each depth's
chosen split: every node's r, m and b, each image's side and next node, each next node's parent, side and key),
records.json (every network's and depth's numbers; "kept" the depths in the tree) and the plot run.png. Each
network logs a line (epochs, its start, train and val mse, gains, seconds split into training, val and the
rest), each depth a summary line with train, val and test mse and the GPU's utilization.

Run from the repo: python train_encoder_components.py --dataset cifar10 --tf32 --bf16
Next: train_prior.py and train_decoder.py on --tree artifacts/<job>.
"""

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # (deep trees fragment the cache)

import torch  # noqa: E402

from dvae.artifacts import artifact  # noqa: E402
from dvae.components import (ImageStream, StepNet, Utilization, answers, best_count, descend, epoch_batches,  # noqa: E402
                             group_state, load_group, mse_now, next_hyperplane, node_loss, orient, root_group,
                             side_stats, split_mse, top_direction)
from dvae.data import SHAPES, get_device, load_images  # noqa: E402
from dvae.deltatree import pixels  # noqa: E402
from dvae.logs import RunLog  # noqa: E402


def atomic_save(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def link(src, dst):
    """dst as a hard link to src (a reused file costs no space), else a copy."""
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def model_path(folder, d, t):
    return os.path.join(folder, "models", f"d{d:03d}_s{t:02d}.pt")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depth-threshold", type=float, default=5e-5, help="val mse per pixel a depth must buy")
    p.add_argument("--lloyd-threshold", type=float, default=None,
                   help="val mse per pixel a Lloyd step must buy (default: the depth threshold / 25)")
    p.add_argument("--epoch-threshold", type=float, default=None,
                   help="val mse per pixel an epoch must buy (default: the Lloyd threshold / 40)")
    p.add_argument("--depth-patience", type=int, default=2, help="depths tried past the best before stopping")
    p.add_argument("--lloyd-patience", type=int, default=2, help="steps tried past a depth's best before moving on")
    p.add_argument("--epoch-patience", type=int, default=2, help="epochs trained past a network's best")
    p.add_argument("--init-patience", type=int, default=1,
                   help="epochs trained past the best for a network that starts from the init run's (already trained "
                        "for its step, so it needs less to show that more training does not pay)")
    p.add_argument("--max-depths", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--images-per-epoch", type=int, default=22528,
                   help="training images per epoch, between a network's val measurements (rounded to whole batches)")
    p.add_argument("--init", default=None, metavar="RUN", help="an earlier run's folder to reuse and start from")
    p.add_argument("--resume", action="store_true",
                   help="reuse this run's own folder as --init does (with --init too, that run's networks stay the "
                        "warm-start candidates)")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--anchor", type=float, default=0.0,
                   help="add this times the squared distance of a network's weights from its start (the warm start it "
                        "was copied from) to its loss, per image and pixel: training stays near the chain of networks "
                        "before it (L2-SP)")
    p.add_argument("--dropout", type=float, default=0.0, help="dropout in each network's blocks while training")
    p.add_argument("--side-m-history", action="store_true", help="history tokens also read each step's side and m")
    p.add_argument("--batch", type=int, default=1024, help="training images per optimizer step")
    p.add_argument("--max-nodes", type=int, default=512, help="nodes per forward pass while training")
    p.add_argument("--node-batch", type=int, default=1024, help="nodes per forward pass when measuring")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=20, help="each network's learning-rate warmup")
    p.add_argument("--history-window", type=int, default=8,
                   help="> 0: a node's history is its last this many deltas plus one token for the sum of those "
                        "before (so a network's cost and the tables kept stop growing with depth); 0: all of them")
    p.add_argument("--table-bits", type=int, choices=(8, 16), default=8,
                   help="the history tables' precision: 16 (half) or 8 (int8 with a scale per row: half the memory, "
                        "so a deeper tree stays on the GPU)")
    p.add_argument("--gpu-tables", type=float, default=0.75,
                   help="share of the GPU's memory the history tables may fill; deeper tables go to host memory")
    p.add_argument("--store-train-r", action=argparse.BooleanOptionalAction, default=True,
                   help="keep the train split's r per node in tree/train (what the prior reads; ~0.28 GB per deep "
                        "CIFAR-10 depth); without it the tree keeps train's structure only")
    p.add_argument("--save-models", action=argparse.BooleanOptionalAction, default=True,
                   help="keep every network in models/ (what --init and --resume reuse; ~9 MB each at width 128)")
    p.add_argument("--store-eval-r", action=argparse.BooleanOptionalAction, default=True,
                   help="keep the val and test splits' r per node in tree/ (what the prior and decoder read for them)")
    p.add_argument("--snapshot-every", type=int, default=100,
                   help="save the groups' state (state.pt: every split's remainders, nodes and windowed history "
                        "tables) every this many depths and at the end, so a rerun with --init (or --resume) that "
                        "would replay all of it starts from it at once instead (needs --history-window > 0)")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true",
                   help="float16 autocast with a gradient scaler (for V100s, which have fp16 tensor cores but no bf16)")
    p.add_argument("--hours", type=float, default=0.0, help="stop after this many hours (0: no limit)")
    p.add_argument("--flip", action="store_true",
                   help="train on the training images and their horizontal mirror images (val and test as they are)")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--limit-test", type=int, default=None)
    p.add_argument("--out", default=None, help="the run's folder (default: its artifacts folder)")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="data")
    p.add_argument("--dataset", choices=sorted(SHAPES), default="cifar10")
    args = p.parse_args()
    parser_defaults = {a.dest: a.default for a in p._actions}
    if args.lloyd_threshold is None:  # (epoch << step << depth: what each must buy, in val mse per pixel)
        args.lloyd_threshold = args.depth_threshold / 25
    if args.epoch_threshold is None:
        args.epoch_threshold = args.lloyd_threshold / 40
    out = args.out or os.path.dirname(artifact("run.log"))
    for sub in ("models", "tree/train", "tree/val", "tree/test"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    init = out if args.resume else args.init  # (the run whose networks are reused while the lineage matches)
    warm = args.init or init  # (the run whose networks are warm-start candidates)
    device = get_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    log = RunLog(os.path.join(out, "run.log"), device)
    util = Utilization(device)
    log(f"=== train_encoder_components.py {time.strftime('%Y-%m-%d')} {vars(args)}")
    began = time.time()

    old = None
    if init is not None:
        with open(os.path.join(init, "records.json")) as f:
            old = json.load(f)
        log(f"reusing {init}: {len(old['models'])} networks, {len(old['depths'])} depths")
    # (a network is reused as it is only if it would see exactly what it was trained on)
    same = ("dataset", "width", "layers", "heads", "side_m_history", "table_bits", "history_window", "limit_train",
            "limit_val", "seed", "flip")
    if old is not None:
        differ = [k for k in same if old["args"].get(k, parser_defaults.get(k)) != getattr(args, k)]
        if differ:
            log(f"{init} differs in {', '.join(differ)}: its networks are warm starts only, none reused as they are")
            old["models"], old["depths"] = [], []
    old_models = {(r["d"], r["t"]): r for r in old["models"]} if old else {}
    old_depths = {r["d"]: r for r in old["depths"]} if old else {}

    images = {s: load_images(args.dataset, s, args.data_root)[:lim]
              for s, lim in (("train", args.limit_train), ("val", args.limit_val), ("test", args.limit_test))}
    if args.flip:  # (the training images and their mirror images: twice the data for every network)
        images["train"] = torch.cat([images["train"], images["train"].flip(-1)])
    data = {s: pixels(x).to(device) for s, x in images.items()}
    mean = data["train"].mean(0)
    groups = {s: root_group(s, X, mean, args.history_window) for s, X in data.items()}
    train, val, test = groups["train"], groups["val"], groups["test"]
    N, D = train.R.shape
    total_memory = torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 0
    budget = args.gpu_tables * total_memory
    shape = SHAPES[args.dataset]
    torch.manual_seed(args.seed)

    def new_net():
        net = StepNet(args.width, args.layers, args.heads, not args.side_m_history, shape, args.dropout).to(device)
        net.bf16, net.fp16 = args.bf16, args.fp16
        return net

    net = new_net()
    log(f"{N} train, {len(val.R)} val, {len(test.R)} test images of {D} pixels; each network "
        f"{sum(q.numel() for q in net.parameters()) / 1e6:.2f}M parameters; history tables on the GPU up to "
        f"{budget / 2 ** 30:.1f} GB")
    root_val = mse_now(val)
    records: dict = dict(args=vars(args), mean=mean.cpu().tolist(), models=[], depths=[], kept=0,
                         root=dict(train=mse_now(train), val=root_val, test=mse_now(test)))
    def save_records():
        atomic_json(records, os.path.join(out, "records.json"))

    drawing: list[subprocess.Popen] = []  # (the redraw going, if any)

    def plot(final=False):
        """Redraw run.png from records.json in a process of its own (a thread would hold this one's interpreter lock
        for seconds once the records are long, starving the GPU). A redraw still going skips this one; final waits
        and draws."""
        if drawing and (drawing[0].poll() is None and not final):
            return
        if drawing:
            if drawing[0].wait() != 0:
                log(f"plot failed (exit {drawing[0].returncode})")
            drawing.clear()
        here = os.path.dirname(os.path.abspath(__file__))
        drawing.append(subprocess.Popen([sys.executable, "-m", "dvae.plots", os.path.join(out, "records.json"),
                                         os.path.join(out, "run.png")], cwd=here))
        if final:
            drawing[0].wait()

    recent: dict | None = None  # the most recent network's state (on the GPU)
    prev_depth: dict[int, dict] = {}  # the previous depth's networks' states by step
    exact = old is not None  # every network so far is the init run's, reused as it is
    chosen_steps: list[int] = []
    depth_gains: list[float] = []  # val mse at the root less after each depth
    offset = 0.0  # (--resume: the old run's seconds for the depths replayed)
    gen = torch.Generator().manual_seed(args.seed)
    stream = ImageStream(N, gen)  # (every network's epochs, in turn)
    epoch_steps = max(1, round(args.images_per_epoch / args.batch))

    def save_state(d):
        """state.pt: the state after depth d, what a later run starts from instead of replaying depths 0..d."""
        if args.history_window <= 0:
            return
        t0 = time.time()
        path = os.path.join(out, "state.pt")
        atomic_save(dict(d=d, groups={s: group_state(g) for s, g in groups.items()}, gen=gen.get_state(),
                         stream=stream.state()), path)
        log(f"state after depth {d} saved: {os.path.getsize(path) / 2 ** 30:.1f} GB in {time.time() - t0:.0f}s")

    # (a rerun whose every choice up to the init run's saved state would be the init run's starts from that state:
    # a lower --depth-threshold changes no depth's steps, so its replay would only redo the forward passes)
    first_d = 0
    state_path = os.path.join(init, "state.pt") if init is not None else ""
    if old is not None and old["depths"] and os.path.exists(state_path):
        choice = ("lloyd_threshold", "lloyd_patience", "max_steps")
        differ = [k for k in choice if old["args"].get(k, parser_defaults.get(k)) != getattr(args, k)]
        if differ:
            log(f"{state_path} unused: the run differs in {', '.join(differ)}, so its steps would be chosen anew")
        else:
            assert init is not None
            t0 = time.time()
            snap = torch.load(state_path, map_location="cpu", weights_only=False)
            last = snap["d"]
            groups.clear()
            torch.cuda.empty_cache() if device.type == "cuda" else None
            groups.update({s: load_group(state, device, budget) for s, state in snap["groups"].items()})
            train, val, test = groups["train"], groups["val"], groups["test"]
            gen.set_state(snap["gen"])
            if "stream" in snap:
                stream.load(snap["stream"])
            for d in range(last + 1):
                r = old_depths[d]
                chosen_steps.append(r["steps"])
                depth_gains.append(old["root"]["val"] - r["val"])
                records["depths"].append(dict(r))
                for t in range(r["tried"]):
                    records["models"].append(dict(old_models[(d, t)], reused=True))
                    if args.save_models and init != out:
                        link(model_path(init, d, t), model_path(out, d, t))
                if init != out:
                    for s in groups:
                        link(os.path.join(init, "tree", s, f"d{d:03d}.pt"), os.path.join(out, "tree", s, f"d{d:03d}.pt"))
            tried = old_depths[last]["tried"]
            prev_depth = {t: torch.load(model_path(init, last, t), map_location=device, weights_only=False)
                          ["state_dict"] for t in range(tried)}
            recent = prev_depth[tried - 1]
            records["kept"] = best_count(depth_gains, args.depth_threshold)
            first_d = last + 1
            offset = old_depths[last]["seconds"] - (time.time() - began)  # (the init run's clock goes on)
            save_records()
            log(f"started from {state_path}: depths 0..{last} as {init} left them (val mse "
                f"{old_depths[last]['val']:.6f}), in {time.time() - t0:.0f}s")

    def val_mse(model, s_va, hv: tuple[torch.Tensor, torch.Tensor]):
        r, m = answers(model, val, hv[0], hv[1], args.node_batch)
        return split_mse(s_va, r, m)

    def train_net(model, s_tr, s_va, hyper_tr, hyper_va, v0, timing, label, patience):
        """Train model on the step's train nodes from val mse v0, keeping its best epoch's weights. Returns
        (epochs run, the kept epoch, the kept val mse)."""
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup_steps))
        best = (0.0, 0, v0, copy.deepcopy(model.state_dict()))  # (objective, epoch, val mse, weights)
        anchors = [(q, q.detach().clone()) for q in model.parameters()] if args.anchor > 0 else []
        scaler = torch.amp.GradScaler(device.type, enabled=args.fp16)  # (float16's gradients would underflow)
        e = 0
        while True:
            t0 = time.time()
            model.train()
            for pieces, B in epoch_batches(train, stream.take(epoch_steps * args.batch), args.batch, args.max_nodes):
                opt.zero_grad(set_to_none=True)
                for nodes, w in pieces:
                    loss = (node_loss(model, train, s_tr, nodes, *hyper_tr) * w).sum() / (B * D)
                    scaler.scale(loss).backward()
                if anchors:
                    pull = sum((q - q0).square().sum() for q, q0 in anchors)
                    assert isinstance(pull, torch.Tensor)
                    scaler.scale(args.anchor * pull).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
            e += 1
            t1 = time.time()
            timing["train"] += t1 - t0
            v = val_mse(model, s_va, hyper_va)
            objective = (v0 - v) - args.epoch_threshold * e
            if objective > best[0]:
                best = (objective, e, v, copy.deepcopy(model.state_dict()))
            log.detail(f"      {label} epoch {e}: val mse {v:.6f} (gain {v0 - v:.2e}, objective {objective:.2e}; "
                       f"best epoch {best[1]})")
            timing["val"] += time.time() - t1
            if e - best[1] >= patience or e >= args.max_epochs:
                break
        model.load_state_dict(best[3])
        return e, best[1], best[2]

    d_done = first_d - 1
    for d in range(first_d, args.max_depths):
        depth_began = time.time()
        timing_depth = dict(train=0.0, val=0.0, other=0.0)
        t_other = time.time()
        start = top_direction(train.R)
        before = dict(train=mse_now(train), val=mse_now(val))
        lineage = ",".join(map(str, chosen_steps))
        hyper = {s: (start.expand(groups[s].nodes, -1), train.R.new_zeros(groups[s].nodes)) for s in ("train", "val")}
        timing_depth["other"] += time.time() - t_other
        gains: list[float] = []
        states: dict[int, dict] = {}
        best_n = 0
        best_split: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
        first_trained = None
        exact_step = exact
        for t in range(args.max_steps):
            timing = dict(train=0.0, val=0.0, other=0.0)
            t_other = time.time()
            mine = model_path(out, d, t)
            prior_rec = old_models.get((d, t))
            reused = exact_step and prior_rec is not None and prior_rec["lineage"] == lineage
            if reused:
                assert init is not None and prior_rec is not None
                saved = torch.load(model_path(init, d, t), map_location=device, weights_only=False)
                net.load_state_dict(saved["state_dict"])
                if t == 0:  # (the start's sign the network was trained with)
                    start = orient(net, train, start, prior_rec["train"], args.node_batch)
                    hyper = {s: (start.expand(groups[s].nodes, -1), train.R.new_zeros(groups[s].nodes))
                             for s in ("train", "val")}
            s_tr, s_va = side_stats(train, *hyper["train"]), side_stats(val, *hyper["val"])
            if reused and prior_rec is not None:
                epochs, kept_epoch, start_label = prior_rec["epochs"], prior_rec["kept_epoch"], "reused"
                v0 = math.nan
                timing["other"] += time.time() - t_other
            else:
                exact_step = False
                first_trained = t if first_trained is None else first_trained
                candidates: list[tuple[str, dict]] = []
                if recent is not None:
                    candidates.append(("most recent", recent))
                if t in prev_depth:
                    candidates.append((f"d{d - 1}s{t}", prev_depth[t]))
                if warm is not None and os.path.exists(model_path(warm, d, t)):
                    saved = torch.load(model_path(warm, d, t), map_location=device, weights_only=False)
                    if saved["config"] == net.config:  # (another architecture's weights cannot start this one)
                        candidates.append(("init run's", saved["state_dict"]))
                timing["other"] += time.time() - t_other
                t_val = time.time()
                if not candidates:
                    start_label, v0 = "fresh", val_mse(net, s_va, hyper["val"])
                else:
                    scored = []
                    for label, state in candidates:
                        net.load_state_dict(state)
                        scored.append((val_mse(net, s_va, hyper["val"]), label, state))
                    v0, start_label, state = min(scored, key=lambda x: x[0])
                    net.load_state_dict(state)
                timing["val"] += time.time() - t_val
                patience = args.init_patience if start_label == "init run's" else args.epoch_patience
                epochs, kept_epoch, _ = train_net(net, s_tr, s_va, hyper["train"], hyper["val"], v0, timing,
                                                  f"d{d}s{t}", patience)
            t_other = time.time()
            r_tr, m_tr = answers(net, train, *hyper["train"], args.node_batch)
            r_va, m_va = answers(net, val, *hyper["val"], args.node_batch)
            after = dict(train=split_mse(s_tr, r_tr, m_tr), val=split_mse(s_va, r_va, m_va))
            # (a reused network's step counts as the init run measured it, so a replay's rounding can't flip a choice)
            gains.append(prior_rec["gain"] if reused and prior_rec is not None else before["val"] - after["val"])
            n_best = best_count(gains, args.lloyd_threshold)
            if n_best == t + 1:
                best_n = n_best
                best_split = dict(train=(hyper["train"][0], hyper["train"][1], r_tr, m_tr),
                                  val=(hyper["val"][0], hyper["val"][1], r_va, m_va))
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            states[t] = state
            recent = state
            if args.save_models and reused and init is not None and init != out:
                link(model_path(init, d, t), mine)
            elif args.save_models and not reused:
                atomic_save(dict(config=net.config, state_dict={k: v.cpu() for k, v in state.items()}, d=d, t=t),
                            mine)
            timing["other"] += time.time() - t_other
            seconds = sum(timing.values())
            for k in timing:
                timing_depth[k] += timing[k]
            records["models"] = [r for r in records["models"] if (r["d"], r["t"]) != (d, t)]
            records["models"].append(dict(d=d, t=t, lineage=lineage, start=start_label, epochs=epochs,
                                          kept_epoch=kept_epoch, val_start=v0, val=after["val"],
                                          train=after["train"], gain=gains[-1], seconds=timing, reused=reused))
            step_gain = gains[-1] - (gains[-2] if t > 0 else 0.0)
            how = "reused" if reused else (f"{epochs} epochs (kept {kept_epoch}), from {start_label} "
                                           f"(val {v0:.6f})")
            log(f"  d{d} s{t}: {how}; val mse {after['val']:.6f} (step gain {step_gain:+.2e}, depth gain "
                f"{gains[-1]:.2e}, best {best_n} steps), train {after['train']:.6f}; {seconds:.1f}s: "
                f"{timing['train']:.1f} train, {timing['val']:.1f} val, {timing['other']:.1f} other")
            if t + 1 - best_n >= args.lloyd_patience:
                break
            t_other = time.time()
            hyper = dict(train=next_hyperplane(r_tr, m_tr), val=next_hyperplane(r_va, m_va))
            timing_depth["other"] += time.time() - t_other
        assert best_split is not None
        t_other = time.time()
        # test: replay the chosen steps
        ht = (start.expand(test.nodes, -1), test.R.new_zeros(test.nodes))
        r_te = m_te = test.R.new_zeros(0)
        for t in range(best_n):
            net.load_state_dict(states[t])
            r_te, m_te = answers(net, test, *ht, args.node_batch)
            if t + 1 < best_n:
                ht = next_hyperplane(r_te, m_te)
        splits: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = dict(
            train=best_split["train"], val=best_split["val"], test=(ht[0], ht[1], r_te, m_te))
        nodes = {}
        for s, (n, b, r, m) in splits.items():
            g = groups[s]
            nodes[s] = g.nodes
            kept = descend(g, n, b, r, m, budget, args.table_bits)
            keep_r = args.store_train_r if s == "train" else args.store_eval_r
            atomic_save(dict(r=r.half().cpu() if keep_r else None, m=m.cpu(), b=b.cpu(), keys=g.keys, start=start.cpu(),
                             **kept),
                        os.path.join(out, "tree", s, f"d{d:03d}.pt"))
        exact = exact and (first_trained is None or first_trained >= best_n)
        chosen_steps.append(best_n)
        prev_depth = states
        now = dict(train=mse_now(train), val=mse_now(val), test=mse_now(test))
        # (a depth replayed whole counts as the init run measured it, as its steps do)
        replayed = exact and d in old_depths and old is not None
        depth_gains.append(old["root"]["val"] - old_depths[d]["val"] if replayed and old is not None
                           else root_val - now["val"])
        k_best = best_count(depth_gains, args.depth_threshold)
        timing_depth["other"] += time.time() - t_other
        if args.resume and exact and d in old_depths:  # (replayed: the old run's clock)
            offset = old_depths[d]["seconds"] - (time.time() - began)
        elapsed = time.time() - began + offset
        busy = util.mean(depth_began)
        records["depths"] = [r for r in records["depths"] if r["d"] != d]
        records["depths"].append(dict(d=d, steps=best_n, tried=len(gains), gains=gains, val=now["val"],
                                      train=now["train"], test=now["test"], gain=depth_gains[-1] - (
                                          depth_gains[-2] if d > 0 else 0.0), nodes=nodes, seconds=elapsed,
                                      wall=time.time() - depth_began, timing=timing_depth, util=busy))
        records["kept"] = k_best
        save_records()
        plot()
        wall = time.time() - depth_began
        log(f"depth {d}: {best_n} of {len(gains)} Lloyd steps; mse val {now['val']:.6f} (gain "
            f"{records['depths'][-1]['gain']:.2e}; best {k_best} depths), train {now['train']:.6f}, test "
            f"{now['test']:.6f}; nodes {nodes['train']} train, {nodes['val']} val; {wall:.0f}s: "
            f"{timing_depth['train']:.0f} train, {timing_depth['val']:.0f} val, {timing_depth['other']:.0f} other"
            + (f"; GPU {busy:.0f}% busy" if busy is not None else "") + f"; {elapsed / 3600:.2f} h in all")
        d_done = d
        if args.snapshot_every > 0 and (d + 1) % args.snapshot_every == 0:
            save_state(d)
        if d + 1 - k_best >= args.depth_patience:
            log(f"no depth past {k_best} paid {args.depth_threshold:.1e} for {args.depth_patience} depths: done")
            break
        if args.hours > 0 and time.time() - began > args.hours * 3600:
            log(f"--hours {args.hours} reached")
            break
    if d_done >= first_d and (args.snapshot_every <= 0 or (d_done + 1) % args.snapshot_every != 0):
        save_state(d_done)
    plot(final=True)
    log(f"the tree keeps {records['kept']} depths: val mse {records['depths'][records['kept'] - 1]['val']:.6f}, "
        f"test {records['depths'][records['kept'] - 1]['test']:.6f}; {out}")


if __name__ == "__main__":
    main()
