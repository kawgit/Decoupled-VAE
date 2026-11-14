"""A prior over a V-way tree's codes (dvae/vprior.py's CandNet: at each depth, an image's delta history as memory and
the tree's V candidates there as queries, as the tree's own networks read them), trained with a discriminator of the
same shape: the prior's loss is a * cross-entropy - lam * E_p[log trust], the discriminator's the usual one (the
real token trusted, the prior's tokens weighted by p not), both updated on every batch. a = 1, lam = 0 is plain
cross-entropy (the discriminator then only scores); a = 0 trains on expected log trust alone.

The candidates and histories come from replaying the tree's saved networks (dvae/vreplay.py) along each image's
code, its first --depths depths, into CandStores on the GPU: a rolling buffer of --buffer training images (the outer
run's training latents in a fixed random order, cycled), and --val val and --test test images, replayed once. A
batch is --batch (image, depth) pairs drawn at random from the buffer. An epoch: the buffer's --rotate oldest images
make way for the next ones (scored before any training on them: "incoming"), --steps batches, then val. Every epoch
logs, over val's every (image, depth): the prior's negative log trust under the discriminator (early stopping keeps
the epoch where it is lowest), its cross-entropy in bits per image and the discriminator's own loss; beside the
baselines: the tree's shares, the best constant share of each token per depth (fit on the first buffer) and
uniform. At the end a fresh discriminator, with the same settings, trains against the kept prior frozen
(early-stopped on its val loss): its negative log trust on val and test is the run's score.

run.png (redrawn every epoch) plots val's negative log trust, bits per image and the discriminator's loss by epoch,
the kept prior's bits saved per token by depth, and QA images (dvae/vprior.py's CodeQA, every --qa-every epochs and
for the kept prior): test images, their reconstructions by the tree, and codes sampled from the prior and from the
tree's shares, each decoded by the tree's networks and the outer decoder.

What the run keeps (artifacts/<job>/): run.log (and run.full.log, with progress within phases), records.json,
run.png, prior.pt (the kept prior), disc.pt (the fresh discriminator).

Run from the repo: python train_vprior.py --tree artifacts/<inner job> --lam 1
"""

import argparse
import copy
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from dvae.artifacts import artifact
from dvae.data import get_device
from dvae.logs import RunLog
from dvae.plots import plot_vprior
from dvae.vprior import CandNet, CandStore, CodeQA
from dvae.vreplay import VTree

LN2 = math.log(2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tree", required=True, metavar="RUN", help="a train_inner_vway.py run that saved its networks")
    p.add_argument("--depths", type=int, default=None, help="K: the codes' first K depths (default: every depth saved)")
    p.add_argument("--a", type=float, default=1.0, help="the weight of cross-entropy in the prior's loss")
    p.add_argument("--lam", type=float, default=0.0, help="the weight of -E_p[log trust] in the prior's loss")
    p.add_argument("--window", type=int, default=None, help="history deltas read (default: the tree's window)")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch", type=int, default=512, help="(image, depth) pairs")
    p.add_argument("--steps", type=int, default=1000, help="batches per epoch")
    p.add_argument("--buffer", type=int, default=4000, help="training images replayed and held on the GPU")
    p.add_argument("--rotate", type=int, default=400, help="buffer images replaced each epoch (0: a fixed buffer)")
    p.add_argument("--val", type=int, default=1000, help="val images scored")
    p.add_argument("--test", type=int, default=1000, help="test images scored at the end")
    p.add_argument("--lr", type=float, default=3e-4, help="both networks'")
    p.add_argument("--beta1", type=float, default=0.5, help="both networks' Adam beta1")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=5, help="epochs without a lower val negative log trust")
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--progress-every", type=int, default=200, help="batches between progress lines (run.full.log)")
    p.add_argument("--chunk", type=int, default=1000, help="images replayed at once")
    p.add_argument("--node-batch", type=int, default=2048)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--qa-n", type=int, default=10, help="QA images of each kind in run.png")
    p.add_argument("--qa-every", type=int, default=5,
                   help="epochs between QA samples (and the kept prior's at the end)")
    p.add_argument("--data-root", default="data")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or os.path.dirname(artifact("run.log"))
    os.makedirs(out, exist_ok=True)
    device = get_device(args.device)
    log = RunLog(os.path.join(out, "run.log"), device)
    log(f"=== train_vprior.py {time.strftime('%Y-%m-%d')} {vars(args)}")
    torch.manual_seed(args.seed)
    began = time.time()

    def amp():
        return torch.autocast(device.type, dtype=torch.float16 if args.fp16 else torch.bfloat16,
                              enabled=args.fp16 or args.bf16)

    t0 = time.time()
    tree = VTree(args.tree, device, args.depths, fp16=args.fp16, node_batch=args.node_batch)
    K, V = tree.K, tree.args["V"]
    W = args.window or tree.window
    qa = CodeQA(tree, args.qa_n, args.data_root, args.seed)
    lat = qa.lat
    D = lat["train"][0].numel()
    log(f"tree {args.tree}: {K} depths ({K * math.log2(V):.0f} bits), V = {V}, latents of {D} numbers, "
        f"{sum(len(r) for r in tree.nets)} networks; history window {W}; loaded with the QA images in "
        f"{time.time() - t0:.0f}s")

    pool = lat["train"].flatten(1)
    order = torch.randperm(len(pool), generator=torch.Generator().manual_seed(args.seed + 2))
    drawn = [0]

    def take(n):
        idx = order[(drawn[0] + torch.arange(n)) % len(pool)]
        drawn[0] += n
        return idx

    def fill(store, rows, Z, label):
        t0 = time.time()
        for i in range(0, len(rows), args.chunk):
            tree.code(Z[i:i + args.chunk].to(device).float(), K, visit=store.writer(rows[i:i + args.chunk]))
            done = min(i + args.chunk, len(rows))
            log.detail(f"    {label}: replayed {done} of {len(rows)} images; "
                       f"{(len(rows) - done) * (time.time() - t0) / done:.0f}s left")
        return time.time() - t0

    stores = {}
    for name, n, Z in (("buffer", args.buffer, None), ("val", args.val, lat["val"]), ("test", args.test, lat["test"])):
        stores[name] = CandStore(n, K, V, D, device)
        Z = pool[take(n)] if Z is None else Z[:n].flatten(1)
        seconds = fill(stores[name], torch.arange(n, device=device), Z, name)
        log(f"{name}: {n} images replayed in {seconds:.0f}s ({stores[name].nbytes() / 2**30:.1f} GiB on the GPU)")
    buf = stores["buffer"]

    # (baselines, bits per token by depth: the tree's own shares, and the best constant share of each token per depth)
    tok0 = buf.tok.long()
    counts = torch.zeros(K, V, device=device).scatter_add_(1, tok0.T, torch.ones_like(tok0.T, dtype=torch.float))
    constant = ((counts + 0.5) / (counts + 0.5).sum(-1, keepdim=True)).log()

    def baseline(store):
        tok = store.tok.long()
        own = -store.share.float().clamp_min(1e-6).log().gather(-1, tok[..., None])[..., 0].mean(0) / LN2
        const = -constant[torch.arange(K, device=device), tok].mean(0) / LN2
        return dict(tree=own.tolist(), constant=const.tolist())
    base_depth = {s: baseline(stores[s]) for s in ("val", "test")}
    baselines = {s: {k: sum(v) for k, v in b.items()} for s, b in base_depth.items()}
    log(f"baselines, bits per image: tree's shares val {baselines['val']['tree']:.2f}, test "
        f"{baselines['test']['tree']:.2f}; constant per depth val {baselines['val']['constant']:.2f}, test "
        f"{baselines['test']['constant']:.2f} (uniform: {K * math.log2(V):.0f})")

    def make():
        return CandNet(K, V, D, W, args.width, args.layers, args.heads, args.dropout).to(device)

    def optimizer(net):
        return torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(args.beta1, 0.999),
                                 weight_decay=args.weight_decay, fused=device.type == "cuda")

    prior, disc = make(), make()
    log(f"prior and discriminator: {sum(q.numel() for q in prior.parameters()) / 1e6:.2f}M parameters each; prior "
        f"loss {args.a:g} * CE - {args.lam:g} * E_p[log trust]; {args.batch} (image, depth) pairs a batch, "
        f"{args.steps} batches an epoch")

    def disc_loss(s, pp, tok):
        """-log trust(real) - sum_v p_v log(1 - trust_v), per pair [B]."""
        return F.softplus(-s.gather(-1, tok[:, None])[:, 0]) + (pp * F.softplus(s)).sum(-1)

    @torch.no_grad()
    def evaluate(store, rows, net_p, net_d, batch=4096):
        """Over every depth of the images rows of store: the prior's cross-entropy (bits per image, and by depth),
        its negative log trust under net_d and net_d's own loss (nats, averaged over images and depths)."""
        net_p.eval()
        net_d.eval()
        ce = torch.zeros(K, device=device)
        nlt = dl = 0.0
        per = max(1, batch // K)
        for s in range(0, len(rows), per):
            r = rows[s:s + per]
            i, d = r.repeat_interleave(K), torch.arange(K, device=device).repeat(len(r))
            inp = store.inputs(i, d, W)
            t = store.tok[i, d].long()
            with amp():
                lp, sd = net_p(*inp).float(), net_d(*inp).float()
            logp = lp.log_softmax(-1)
            pp = logp.exp()
            ce.index_add_(0, d, -logp.gather(-1, t[:, None])[:, 0])
            nlt -= (pp * F.logsigmoid(sd)).sum().item()
            dl += disc_loss(sd, pp, t).sum().item()
        n = len(rows)
        by_depth = (ce / n / LN2).tolist()
        return dict(bits=sum(by_depth), nlt=nlt / (n * K), disc=dl / (n * K)), by_depth

    records: dict = dict(args=vars(args), K=K, V=V, window=W, baselines=baselines, history=[], fresh=[], final=None)
    samples: dict = {}
    oldest = [0]

    def save():
        atomic_json(records, os.path.join(out, "records.json"))
        plot_vprior(records, os.path.join(out, "run.png"), samples or None)

    def train_epochs(net_p, net_d, opt_p, opt_d, train_prior, label, history):
        """Epochs until the selection metric (train_prior: val negative log trust; else the discriminator's val
        loss) passes --patience without a new low; returns the best (metric, epoch, prior state, disc state)."""
        sc_p = torch.amp.GradScaler(device.type, enabled=args.fp16)
        sc_d = torch.amp.GradScaler(device.type, enabled=args.fp16)
        val_rows = torch.arange(args.val, device=device)
        best = (float("inf"), 0, None, None)
        for e in range(1, args.max_epochs + 1):
            t0 = time.time()
            timing = {}
            incoming = None
            if args.rotate and (e > 1 or history is records["fresh"]):  # (the oldest make way, scored first)
                rows = (oldest[0] + torch.arange(args.rotate, device=device)) % args.buffer
                oldest[0] = (oldest[0] + args.rotate) % args.buffer
                fill(buf, rows, pool[take(args.rotate)], "rotation")
                incoming = evaluate(buf, rows, net_p, net_d)[0]
                timing["rotate"] = time.time() - t0
            t1 = time.time()
            net_p.train(train_prior)
            net_d.train()
            sums = dict(ce=0.0, ns=0.0, dl=0.0)
            for b in range(args.steps):
                i = torch.randint(0, args.buffer, (args.batch,), device=device)
                d = torch.randint(0, K, (args.batch,), device=device)
                inp = buf.inputs(i, d, W)
                t = buf.tok[i, d].long()
                with amp():
                    with torch.set_grad_enabled(train_prior):  # (a frozen prior builds no graph)
                        lp = net_p(*inp).float()
                    sd = net_d(*inp).float()
                logp = lp.log_softmax(-1)
                pp = logp.exp()
                ld = disc_loss(sd, pp.detach(), t).mean()
                opt_d.zero_grad(set_to_none=True)
                sc_d.scale(ld).backward()
                sc_d.step(opt_d)
                sc_d.update()
                ce = -logp.gather(-1, t[:, None]).mean()
                ns = -(pp * F.logsigmoid(sd.detach())).sum(-1).mean()
                if train_prior:
                    assert opt_p is not None
                    lpr = args.a * ce + args.lam * ns
                    opt_p.zero_grad(set_to_none=True)
                    sc_p.scale(lpr).backward()
                    sc_p.step(opt_p)
                    sc_p.update()
                sums["ce"] += ce.item()
                sums["ns"] += ns.item()
                sums["dl"] += ld.item()
                if (b + 1) % args.progress_every == 0:
                    rate = (b + 1) / (time.time() - t1)
                    log.detail(f"    {label} epoch {e}: batch {b + 1} of {args.steps}; train CE "
                               f"{sums['ce'] / (b + 1) * K / LN2:.2f} bits per image, -E_p[log trust] "
                               f"{sums['ns'] / (b + 1):.4f}, disc loss {sums['dl'] / (b + 1):.4f}; {rate:.1f} "
                               f"batches/s, {(args.steps - b - 1) / rate:.0f}s left")
            timing["train"] = time.time() - t1
            t1 = time.time()
            v, v_depth = evaluate(stores["val"], val_rows, net_p, net_d)
            timing["val"] = time.time() - t1
            metric = v["nlt"] if train_prior else v["disc"]
            if metric < best[0]:
                best = (metric, e, copy.deepcopy(net_p.state_dict()), copy.deepcopy(net_d.state_dict()))
                if train_prior:
                    records["by_depth"] = dict(epoch=e, prior=v_depth, **base_depth["val"])
            if train_prior and (e == 1 or e % args.qa_every == 0):
                t1 = time.time()
                samples.update(qa.samples(net_p, amp), epoch=e)
                timing["samples"] = time.time() - t1
            history.append(dict(epoch=e, train_bits=sums["ce"] / args.steps * K / LN2,
                                train_nlt=sums["ns"] / args.steps, train_disc=sums["dl"] / args.steps,
                                incoming=incoming, val=v, timing=timing, seconds=time.time() - t0,
                                hours=(time.time() - began) / 3600))
            seen = f"incoming {incoming['bits']:.2f}, " if incoming else ""
            log(f"{label} epoch {e}: val -E_p[log trust] {v['nlt']:.4f} (ln 2 = {LN2:.4f}), bits per image "
                f"{v['bits']:.2f} ({seen}train {history[-1]['train_bits']:.2f}; tree's shares "
                f"{baselines['val']['tree']:.2f}, constant {baselines['val']['constant']:.2f}), disc loss "
                f"{v['disc']:.4f}; best epoch {best[1]}; {time.time() - t0:.0f}s ("
                + ", ".join(f"{k} {s:.0f}" for k, s in timing.items()) + ")")
            save()
            if e - best[1] >= args.patience:
                break
        return best

    best = train_epochs(prior, disc, optimizer(prior), optimizer(disc), True, "prior", records["history"])
    assert best[2] is not None, "no epoch ran"
    prior.load_state_dict(best[2])
    log(f"kept the prior of epoch {best[1]} (val -E_p[log trust] {best[0]:.4f}); a fresh discriminator now trains "
        f"against it, frozen")
    torch.save(dict(config=prior.config, state_dict=prior.state_dict(), epoch=best[1], args=vars(args)),
               os.path.join(out, "prior.pt"))
    samples.update(qa.samples(prior, amp), epoch=best[1])
    judge = make()
    best_j = train_epochs(prior, judge, None, optimizer(judge), False, "fresh discriminator", records["fresh"])
    assert best_j[3] is not None, "no epoch ran"
    judge.load_state_dict(best_j[3])
    torch.save(dict(config=judge.config, state_dict=judge.state_dict(), epoch=best_j[1]), os.path.join(out, "disc.pt"))
    final = {s: evaluate(stores[s], torch.arange(stores[s].n, device=device), prior, judge)[0]
             for s in ("val", "test")}
    records["final"] = final
    log(f"score (the fresh discriminator, epoch {best_j[1]}): -E_p[log trust] val {final['val']['nlt']:.4f}, test "
        f"{final['test']['nlt']:.4f}; bits per image val {final['val']['bits']:.2f}, test {final['test']['bits']:.2f} "
        f"(tree's shares {baselines['test']['tree']:.2f}, constant {baselines['test']['constant']:.2f})")
    save()
    log(f"done in {(time.time() - began) / 60:.1f} min; {out}")


def atomic_json(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
