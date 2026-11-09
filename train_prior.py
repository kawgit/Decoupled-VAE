"""Stage 3: train the prior (dvae/prior.py's TokenPrior), a causal transformer over the encoder's codes that
predicts each token (one of V children), on the codes dump_encoder.py wrote.

The prior reads each node's children (their centroids) and the tokens before, not the shares the encoder's
centroids imply (dvae/refine.py), and starts uniform; it is trained on cross-entropy alone. Each step takes a batch of
training codes and minimizes the cross-entropy of every token, all depths at once. Every epoch logs, in bits, the
train codes' cost (a fixed subset, scored as val is) and the val codes' cost per image and per range of depths,
beside two baselines the prior has to beat: the encoder's implied shares and the best constant share of each token
per depth (fit on the train codes). Early stopping watches the val cost separately in each of
--bands bands of depths (equal widths), since depths need different amounts of training: each band keeps the
weights of the epoch with its lowest val cost, stops once it goes a --patience-frac share of the run's epochs (at least --patience) without
a better one (dvae/stopping.py; the model still trains on every depth), and the run ends when every band has stopped. The kept prior answers each depth
with its band's weights (dvae/prior.py's BandedPrior; --bands 1 is one early stop for all), and at the end it is
scored on the test codes against the same baselines. The progress plot (dvae/plots.py's plot_prior) is redrawn
every epoch.

Run from the repo: python train_prior.py --codes artifacts/<dump job> --images
Next: qa_prior.py samples from it; train_decoder.py can price its codes with it.
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F

from dvae.artifacts import artifact
from dvae.codes import TokenCodes, load_token_codes, own_bits
from dvae.data import get_device
from dvae.logs import RunLog
from dvae.plots import plot_prior
from dvae.prior import TokenPrior, band_edges, code_token_bits, load_token_prior
from dvae.stopping import add_patience_args, patience, stagnant

LN2 = math.log(2)


def bands(K):
    return [(lo, min(hi, K)) for lo, hi in ((1, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 256)) if lo <= K]


def describe(per_token):
    """Bits per image, then bits per token over each range of depths."""
    return f"{per_token.sum():.3f} bits/image (per token " + ", ".join(
        f"{lo}-{hi}: {per_token[lo - 1:hi].mean():.4f}" for lo, hi in bands(len(per_token))) + ")"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--codes", required=True, help="folder with dump_encoder.py's train.pt, val.pt and test.pt")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--images", action="store_true",
                   help="also show each token the children's centroids it chooses between, as images (dvae/prior.py)")
    p.add_argument("--wd", type=float, default=0.05, help="AdamW weight decay (on weight matrices only)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500, help="steps of linear learning-rate warmup")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1000)
    add_patience_args(p)
    p.add_argument("--bands", type=int, default=8, help="bands of depths, each early-stopped on its own")
    p.add_argument("--train-eval", type=int, default=5000, help="train codes scored each epoch, beside val")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None, help="checkpoint (default: the run's artifacts folder's prior.pt)")
    p.add_argument("--log", default=None, help="log (default: the run's artifacts folder's run.log)")
    p.add_argument("--plot", default=None, help="progress plot, redrawn every epoch (default: --log's path as .png)")
    args = p.parse_args()
    args.out, args.log = args.out or artifact("prior.pt"), args.log or artifact("run.log")
    args.plot = args.plot or os.path.splitext(args.log)[0] + ".png"
    device = get_device(args.device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    log = RunLog(args.log, device)
    log(f"=== train_prior.py {vars(args)}")

    began = time.time()
    train, val, test = (load_token_codes(os.path.join(args.codes, f"{s}.pt")) for s in ("train", "val", "test"))
    N, K, V = len(train), train.depths, train.V
    log(f"codes: {N} train, {len(val)} val, {len(test)} test, {K} tokens of {V} children, from {train.ckpt} "
        f"({time.time() - began:.0f}s to load)")

    freq = torch.zeros(K, V).scatter_add_(1, train.tokens.T, torch.ones(K, N)) / N  # each token's share per depth

    def baselines(codes: TokenCodes):
        constant = -freq.clamp_min(1e-6).log2().gather(1, codes.tokens.T).T
        return own_bits(codes).mean(0), constant.mean(0)

    def cost(model, codes: TokenCodes):
        return code_token_bits(model, codes.offsets, codes.tokens, device, args.batch).mean(0)

    own, constant = baselines(val)
    log(f"val baselines: the encoder's implied shares {describe(own)}")
    log(f"val baselines: constant share per depth {describe(constant)}")

    prior = TokenPrior(K, V, args.width, args.layers, args.heads, args.dropout, images=args.images).to(device)
    if args.images:  # the children's centroids are offsets from the predicted means, which start from the training mean
        prior.mean.copy_(train.mean.to(device))
    decay = [q for q in prior.parameters() if q.dim() >= 2]
    rest = [q for q in prior.parameters() if q.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=args.wd), dict(params=rest, weight_decay=0.0)],
                            lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
    log(f"prior: {sum(q.numel() for q in prior.parameters()) / 1e6:.2f}M parameters")

    gen = torch.Generator().manual_seed(args.seed)
    seen = torch.randperm(N, generator=gen)[:args.train_eval]
    train_seen = train.subset(seen)
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
            tokens = train.tokens[idx].to(device, non_blocking=True)
            logits = prior(train.offsets[idx].to(device, non_blocking=True).float(), tokens)
            loss = F.cross_entropy(logits.flatten(0, 1), tokens.flatten())
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
        tr, va = cost(prior, train_seen), cost(prior, val)
        history.append(dict(epoch=epoch, train_loss=running / N / LN2, train=tr.tolist(), val=va.tolist(),
                            seconds=time.time() - began))
        improved = []
        for b, (lo, hi) in enumerate(spans):
            band_cost = va[lo:hi].sum().item()
            if not stopped[b] and band_cost < best[b][0]:
                best[b] = (band_cost, epoch)
                improved.append(names[b])
        if improved:
            states[epoch] = {k: v.detach().cpu().clone() for k, v in prior.state_dict().items()}
            kept = [e for _, e in best]
            states = {e: w for e, w in states.items() if e in kept}
            torch.save(dict(config=prior.config, bands=edges, band_epochs=kept, states=states, args=vars(args),
                            codes=args.codes, encoder=train.ckpt, epoch=epoch, history=history), args.out)
        stopped = [done or stagnant(epoch, e, args) for done, (_, e) in zip(stopped, best)]
        plot_prior(history, edges, [e for _, e in best], args.plot)
        bands_now = ", ".join(f"{n}@{e}" + (" stopped" if done else "")
                              for n, (_, e), done in zip(names, best, stopped))
        log(f"epoch {epoch}: val {describe(va)}; train {describe(tr)}; training loss "
            f"{running / N / LN2:.4f} bits/token; encoder's implied shares {own.sum():.3f}; kept bands "
            f"{sum(c for c, _ in best):.3f} bits/image ({bands_now})"
            f"{'; better in ' + ', '.join(improved) if improved else ''} ({trained:.0f}s train, "
            f"{time.time() - began - trained:.0f}s eval)")
        if all(stopped):
            log(f"every band went {patience(epoch, args)} epochs or more without a better val cost: stopping")
            break

    kept = load_token_prior(torch.load(args.out, map_location=device, weights_only=False), device)
    own, constant = baselines(test)
    te = cost(kept, test)
    log(f"kept epochs per band {', '.join(f'{n}@{e}' for n, (_, e) in zip(names, best))} (val "
        f"{sum(c for c, _ in best):.3f} bits/image), saved {args.out}")
    log(f"test: prior {describe(te)}")
    log(f"test: encoder's implied shares {describe(own)}")
    log(f"test: constant share per depth {describe(constant)}")


if __name__ == "__main__":
    main()
