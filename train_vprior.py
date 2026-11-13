"""A prior over a V-way tree's codes (dvae/vprior.py's CodeNet), trained with a discriminator: the prior's loss is
a * cross-entropy - lam * E_p[log trust], the discriminator's the usual one (real tokens trusted, the prior's
tokens weighted by p not), both updated on every batch. a = 1, lam = 0 is plain cross-entropy (the discriminator
then only scores); a = 0 trains on expected log trust alone.

The codes come from a train_inner_vway.py run's tree files (tree/<split>/dDDD.pt: each image's token at every
depth, and the tree's shares at its node there), its first --depths depths: no replay. Every epoch logs on val,
over its codes: the prior's negative log trust under the discriminator, averaged over depths (early stopping keeps
the epoch where it is lowest), its cross-entropy in bits per image, and the discriminator's own loss; beside two
baselines, the tree's shares and the best constant share of each token per depth (fit on train). At the end a
fresh discriminator, with the same settings, trains against the kept prior frozen (early-stopped on its val loss):
its negative log trust on val and test is the run's score.

What the run keeps (artifacts/<job>/): run.log (and run.full.log, with progress within epochs), records.json,
prior.pt (the kept prior), disc.pt (the fresh discriminator).

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
from dvae.vprior import CodeNet

LN2 = math.log(2)


def load_codes(run: str, split: str, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """split's codes in run's tree files, first K depths: tokens [N, K] (uint8) and the tree's shares at each
    image's node there [N, K, V] (float16)."""
    tokens, shares, node = [], [], None
    for d in range(K):
        f = torch.load(os.path.join(run, "tree", split, f"d{d:03d}.pt"), weights_only=False)
        tokens.append(f["token"].to(torch.uint8))
        shares.append(f["share"][node if node is not None else torch.zeros(len(f["token"]), dtype=torch.long)])
        node = f["node"].long()
    return torch.stack(tokens, 1), torch.stack(shares, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tree", required=True, metavar="RUN", help="a train_inner_vway.py run (its tree/ files)")
    p.add_argument("--depths", type=int, default=None, help="K: the codes' first K depths (default: every depth saved)")
    p.add_argument("--a", type=float, default=1.0, help="the weight of cross-entropy in the prior's loss")
    p.add_argument("--lam", type=float, default=0.0, help="the weight of -E_p[log trust] in the prior's loss")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4, help="both networks'")
    p.add_argument("--beta1", type=float, default=0.5, help="both networks' Adam beta1")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=5, help="epochs without a lower val negative log trust")
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--progress-every", type=int, default=200, help="batches between progress lines (run.full.log)")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--limit-train", type=int, default=None)
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

    with open(os.path.join(args.tree, "records.json")) as f:
        rec = json.load(f)
    V = rec["args"]["V"]
    saved = min(len(os.listdir(os.path.join(args.tree, "tree", s))) for s in ("train", "val", "test"))
    K = min(args.depths or saved, saved, len(rec["depths"]))
    codes = {}
    for s in ("train", "val", "test"):
        t0 = time.time()
        tok, sh = load_codes(args.tree, s, K)
        if s == "train" and args.limit_train:
            tok, sh = tok[:args.limit_train], sh[:args.limit_train]
        codes[s] = (tok.to(device).long(), sh.float().clamp_min(1e-6).log().half().to(device))
        log(f"{s}: {len(tok)} codes of {K} depths ({K * math.log2(V):.0f} bits) loaded in {time.time() - t0:.0f}s")

    # (baselines, bits per image: the tree's own shares, and the best constant share of each token per depth)
    counts = torch.zeros(K, V, device=device).scatter_add_(1, codes["train"][0].T, torch.ones_like(codes["train"][0].T,
                                                                                                   dtype=torch.float))
    constant = ((counts + 0.5) / (counts + 0.5).sum(-1, keepdim=True)).log()

    def baseline_bits(s):
        tok, ls = codes[s]
        tree = -ls.float().gather(-1, tok[..., None])[..., 0].sum(1).mean().item() / LN2
        const = -constant[torch.arange(K, device=device), tok].sum(1).mean().item() / LN2
        return dict(tree=tree, constant=const)
    baselines = {s: baseline_bits(s) for s in ("val", "test")}
    log(f"baselines, bits per image: tree's shares val {baselines['val']['tree']:.2f}, test "
        f"{baselines['test']['tree']:.2f}; constant per depth val {baselines['val']['constant']:.2f}, test "
        f"{baselines['test']['constant']:.2f} (uniform: {K * math.log2(V):.0f})")

    def make():
        return CodeNet(K, V, args.width, args.layers, args.heads, args.dropout).to(device)

    def optimizer(net):
        return torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(args.beta1, 0.999),
                                 weight_decay=args.weight_decay, fused=device.type == "cuda")

    def amp():
        return torch.autocast(device.type, dtype=torch.float16 if args.fp16 else torch.bfloat16,
                              enabled=args.fp16 or args.bf16)

    prior, disc = make(), make()
    log(f"prior and discriminator: {sum(q.numel() for q in prior.parameters()) / 1e6:.2f}M parameters each; prior "
        f"loss {args.a:g} * CE - {args.lam:g} * E_p[log trust]")

    def disc_loss(s_logits, p, tok):
        """-log trust(real) - sum_v p_v log(1 - trust_v), per depth [B, K]."""
        return F.softplus(-s_logits.gather(-1, tok[..., None])[..., 0]) + (p * F.softplus(s_logits)).sum(-1)

    @torch.no_grad()
    def evaluate(s, net_p, net_d, batch=256):
        """split s: the prior's cross-entropy (bits per image), its negative log trust under net_d (nats, averaged
        over depths and images) and net_d's own loss (per depth)."""
        net_p.eval()
        net_d.eval()
        tok, ls = codes[s]
        ce = nlt = dl = 0.0
        for i in range(0, len(tok), batch):
            t, l = tok[i:i + batch], ls[i:i + batch]
            with amp():
                lp, sd = net_p(t, l).float(), net_d(t, l).float()
            logp = lp.log_softmax(-1)
            pp = logp.exp()
            ce -= logp.gather(-1, t[..., None]).sum().item()
            nlt -= (pp * F.logsigmoid(sd)).sum().item()
            dl += disc_loss(sd, pp, t).sum().item()
        n = len(tok)
        return dict(bits=ce / n / LN2, nlt=nlt / (n * K), disc=dl / (n * K))

    def train_epochs(net_p, net_d, opt_p, opt_d, train_prior, label, history):
        """Epochs until the selection metric (train_prior: val negative log trust; else the discriminator's val
        loss) passes --patience without a new low; returns the best (metric, epoch, prior state, disc state)."""
        sc_p = torch.amp.GradScaler(device.type, enabled=args.fp16)
        sc_d = torch.amp.GradScaler(device.type, enabled=args.fp16)
        tok_all, ls_all = codes["train"]
        best = (float("inf"), 0, None, None)
        for e in range(1, args.max_epochs + 1):
            t0 = time.time()
            net_p.train(train_prior)
            net_d.train()
            sums = dict(ce=0.0, ns=0.0, dl=0.0)
            order = torch.randperm(len(tok_all), device=device)
            steps = (len(order) + args.batch - 1) // args.batch
            for b, idx in enumerate(order.split(args.batch)):
                t, l = tok_all[idx], ls_all[idx]
                with amp():
                    with torch.set_grad_enabled(train_prior):  # (a frozen prior builds no graph)
                        lp = net_p(t, l).float()
                    sd = net_d(t, l).float()
                logp = lp.log_softmax(-1)
                pp = logp.exp()
                ld = disc_loss(sd, pp.detach(), t).mean()
                opt_d.zero_grad(set_to_none=True)
                sc_d.scale(ld).backward()
                sc_d.step(opt_d)
                sc_d.update()
                ce = -logp.gather(-1, t[..., None]).mean()
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
                    rate = (b + 1) / (time.time() - t0)
                    log.detail(f"    {label} epoch {e}: batch {b + 1} of {steps}; train CE "
                               f"{sums['ce'] / (b + 1) * K / LN2:.2f} bits per image, -E_p[log trust] "
                               f"{sums['ns'] / (b + 1):.4f}, disc loss {sums['dl'] / (b + 1):.4f}; {rate:.1f} batches/s, "
                               f"{(steps - b - 1) / rate:.0f}s left")
            v = evaluate("val", net_p, net_d)
            metric = v["nlt"] if train_prior else v["disc"]
            if metric < best[0]:
                best = (metric, e, copy.deepcopy(net_p.state_dict()), copy.deepcopy(net_d.state_dict()))
            history.append(dict(epoch=e, train_bits=sums["ce"] / steps * K / LN2, train_nlt=sums["ns"] / steps,
                                train_disc=sums["dl"] / steps, val=v, seconds=time.time() - t0,
                                hours=(time.time() - began) / 3600))
            log(f"{label} epoch {e}: val -E_p[log trust] {v['nlt']:.4f} (ln 2 = {LN2:.4f}), bits per image "
                f"{v['bits']:.2f} (tree's shares {baselines['val']['tree']:.2f}), disc loss {v['disc']:.4f}; train "
                f"bits {history[-1]['train_bits']:.2f}; best epoch {best[1]}; {time.time() - t0:.0f}s")
            atomic_json(dict(args=vars(args), K=K, V=V, baselines=baselines, history=records["history"],
                             fresh=records["fresh"], final=records.get("final")), os.path.join(out, "records.json"))
            if e - best[1] >= args.patience:
                break
        return best

    records: dict = dict(history=[], fresh=[])
    best = train_epochs(prior, disc, optimizer(prior), optimizer(disc), True, "prior", records["history"])
    assert best[2] is not None, "no epoch ran"
    prior.load_state_dict(best[2])
    log(f"kept the prior of epoch {best[1]} (val -E_p[log trust] {best[0]:.4f}); a fresh discriminator now trains "
        f"against it, frozen")
    torch.save(dict(config=prior.config, state_dict=prior.state_dict(), epoch=best[1], args=vars(args)),
               os.path.join(out, "prior.pt"))
    judge = make()
    best_j = train_epochs(prior, judge, None, optimizer(judge), False, "fresh discriminator", records["fresh"])
    assert best_j[3] is not None, "no epoch ran"
    judge.load_state_dict(best_j[3])
    torch.save(dict(config=judge.config, state_dict=judge.state_dict(), epoch=best_j[1]), os.path.join(out, "disc.pt"))
    final = {s: evaluate(s, prior, judge) for s in ("val", "test")}
    records["final"] = final
    log(f"score (the fresh discriminator, epoch {best_j[1]}): -E_p[log trust] val {final['val']['nlt']:.4f}, test "
        f"{final['test']['nlt']:.4f}; bits per image val {final['val']['bits']:.2f}, test {final['test']['bits']:.2f} "
        f"(tree's shares {baselines['test']['tree']:.2f}, constant {baselines['test']['constant']:.2f})")
    atomic_json(dict(args=vars(args), K=K, V=V, baselines=baselines, history=records["history"],
                     fresh=records["fresh"], final=final), os.path.join(out, "records.json"))
    log(f"done in {(time.time() - began) / 60:.1f} min; {out}")


def atomic_json(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
