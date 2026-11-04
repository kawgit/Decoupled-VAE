"""Train a prior over a frozen delta tree's codes (dvae/prior.py's SidePrior) on codes dump_codes.py wrote.

Each step takes a batch of training codes and minimizes the cross-entropy of every token's side under the prior,
all depths at once. Every epoch logs, in bits, the train codes' cost (a fixed subset, scored as val is) and the
val codes' cost per image and per range of depths, beside two baselines the prior has to beat: the tree's own
shares (the rare side with probability m / (1 + m), what qa.py samples from without a prior) and the best
constant share per depth (fit on the train codes). Early stopping watches the val cost separately in each of
--bands bands of depths (equal widths), since depths need different amounts of training: each band keeps the
weights of the epoch with its lowest val cost, stops once --patience epochs pass without a better one (the model
still trains on every depth), and the run ends when every band has stopped. The kept prior answers each depth
with its band's weights (dvae/prior.py's BandedPrior; --bands 1 is one early stop for all), and at the end it is
scored on the test codes against the same baselines.

Run from the repo: python train_prior.py --codes artifacts/<dump job>
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F

from dvae.artifacts import artifact
from dvae.data import get_device
from dvae.logs import RunLog
from dvae.prior import SidePrior, band_edges, load_prior

LN2 = math.log(2)


def load_codes(path):
    """(rare, m, r) from a dump_codes.py file, and the tree checkpoint that made them."""
    c = torch.load(path, weights_only=False)
    return (c["rare"], c["m"], c["r"]), c["ckpt"]


def bits_of(p, rare):
    """Bits [N, K] each token costs when it is rare with probability p [N, K]."""
    return -torch.where(rare, p, 1 - p).clamp_min(1e-12).log2()


@torch.no_grad()
def prior_bits(prior, codes, device, batch):
    """The bits each token of codes costs under the prior, averaged over the codes [K]."""
    prior.eval()
    rare, m, r = codes
    total = torch.zeros(rare.shape[1], device=device)
    for i in range(0, len(rare), batch):
        rb = rare[i:i + batch].to(device)
        logits = prior(r[i:i + batch].to(device).float(), m[i:i + batch].to(device).float(), rb)
        total += F.binary_cross_entropy_with_logits(logits, rb.float(), reduction="none").sum(0) / LN2
    return (total / len(rare)).cpu()


def bands(K):
    return [(lo, min(hi, K)) for lo, hi in ((1, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 256)) if lo <= K]


def describe(per_token):
    """Bits per image, then bits per token over each range of depths."""
    return f"{per_token.sum():.3f} bits/image (per token " + ", ".join(
        f"{lo}-{hi}: {per_token[lo - 1:hi].mean():.4f}" for lo, hi in bands(len(per_token))) + ")"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--codes", required=True, help="folder with dump_codes.py's train.pt, val.pt and test.pt")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--images", action="store_true",
                   help="also show each token the two side means it chooses between, as images (dvae/prior.py)")
    p.add_argument("--wd", type=float, default=0.05, help="AdamW weight decay (on weight matrices only)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500, help="steps of linear learning-rate warmup")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=10, help="epochs without a better val cost before a band stops")
    p.add_argument("--bands", type=int, default=8, help="bands of depths, each early-stopped on its own")
    p.add_argument("--train-eval", type=int, default=5000, help="train codes scored each epoch, beside val")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None, help="checkpoint (default: the run's artifacts folder's prior.pt)")
    p.add_argument("--log", default=None, help="log (default: the run's artifacts folder's run.log)")
    args = p.parse_args()
    args.out, args.log = args.out or artifact("prior.pt"), args.log or artifact("run.log")
    device = get_device(args.device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    log = RunLog(args.log, device)
    log(f"=== train_prior.py {vars(args)}")

    began = time.time()
    (train, tree), (val, _), (test, _) = (load_codes(os.path.join(args.codes, f"{s}.pt"))
                                          for s in ("train", "val", "test"))
    N, K = train[0].shape
    log(f"codes: {N} train, {len(val[0])} val, {len(test[0])} test, {K} tokens ({time.time() - began:.0f}s to load)")

    def baselines(codes):
        rare, m, _ = codes
        own = bits_of(m.float() / (1 + m.float()), rare).mean(0)
        share = train[0].float().mean(0).clamp(1e-6, 1 - 1e-6)
        return own, bits_of(share.expand_as(rare), rare).mean(0)

    own, constant = baselines(val)
    log(f"val baselines: the tree's own shares {describe(own)}")
    log(f"val baselines: constant share per depth {describe(constant)}")

    prior = SidePrior(K, args.width, args.layers, args.heads, args.dropout, images=args.images).to(device)
    if args.images:  # the side means start from the tree's training mean
        prior.mean.copy_(torch.load(tree, map_location="cpu", weights_only=False)["mean"].to(device))
    decay = [q for q in prior.parameters() if q.dim() >= 2]
    rest = [q for q in prior.parameters() if q.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=args.wd), dict(params=rest, weight_decay=0.0)],
                            lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
    log(f"prior: {sum(q.numel() for q in prior.parameters()) / 1e6:.2f}M parameters")

    gen = torch.Generator().manual_seed(args.seed)
    seen = torch.randperm(N, generator=gen)[:args.train_eval]
    train_seen = tuple(t[seen] for t in train)
    edges = band_edges(K, args.bands)
    spans = list(zip(edges, edges[1:]))
    names = [f"{lo + 1}-{hi}" for lo, hi in spans]
    log(f"bands of depths: {', '.join(names)}")
    history, step = [], 0
    best = [(math.inf, -1)] * len(spans)  # each band's lowest val cost (bits per image) and its epoch
    stopped = [False] * len(spans)
    states = {}  # the kept epochs' weights, on the CPU
    for epoch in range(args.epochs):
        began = time.time()
        prior.train()
        order = torch.randperm(N, generator=gen)
        running = 0.0
        for i in range(0, N, args.batch):
            idx = order[i:i + args.batch]
            rare = train[0][idx].to(device, non_blocking=True)
            logits = prior(train[2][idx].to(device, non_blocking=True).float(),
                           train[1][idx].to(device, non_blocking=True).float(), rare)
            loss = F.binary_cross_entropy_with_logits(logits, rare.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            running += loss.item() * len(idx)
            if step % 100 == 0:
                log.detail(f"  step {step}: loss {loss.item() / LN2:.4f} bits/token")
        trained = time.time() - began
        tr, va = prior_bits(prior, train_seen, device, args.batch), prior_bits(prior, val, device, args.batch)
        history.append(dict(epoch=epoch, train_loss=running / N / LN2, train=tr.tolist(), val=va.tolist(),
                            seconds=time.time() - began))
        improved = []
        for b, (lo, hi) in enumerate(spans):
            cost = va[lo:hi].sum().item()
            if not stopped[b] and cost < best[b][0]:
                best[b] = (cost, epoch)
                improved.append(names[b])
        if improved:
            states[epoch] = {k: v.detach().cpu().clone() for k, v in prior.state_dict().items()}
            kept = [e for _, e in best]
            states = {e: w for e, w in states.items() if e in kept}
            torch.save(dict(config=prior.config, bands=edges, band_epochs=kept, states=states, args=vars(args),
                            epoch=epoch, history=history), args.out)
        stopped = [done or epoch - e >= args.patience for done, (_, e) in zip(stopped, best)]
        bands_now = ", ".join(f"{n}@{e}" + (" stopped" if done else "")
                              for n, (_, e), done in zip(names, best, stopped))
        log(f"epoch {epoch}: val {describe(va)}; train {describe(tr)}; training loss "
            f"{running / N / LN2:.4f} bits/token; tree's own shares {own.sum():.3f}; kept bands "
            f"{sum(c for c, _ in best):.3f} bits/image ({bands_now})"
            f"{'; better in ' + ', '.join(improved) if improved else ''} ({trained:.0f}s train, "
            f"{time.time() - began - trained:.0f}s eval)")
        if all(stopped):
            log(f"every band went {args.patience} epochs without a better val cost: stopping")
            break

    kept = load_prior(torch.load(args.out, map_location=device, weights_only=False), device)
    own, constant = baselines(test)
    te = prior_bits(kept, test, device, args.batch)
    log(f"kept epochs per band {', '.join(f'{n}@{e}' for n, (_, e) in zip(names, best))} (val "
        f"{sum(c for c, _ in best):.3f} bits/image), saved {args.out}")
    log(f"test: prior {describe(te)}")
    log(f"test: tree's own shares {describe(own)}")
    log(f"test: constant share per depth {describe(constant)}")


if __name__ == "__main__":
    main()
