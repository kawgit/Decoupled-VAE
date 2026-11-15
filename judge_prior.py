"""How well a prior's samples pass for real: a fresh discriminator, trained by cross-entropy to tell real codes from
the prior's, scored by its log trust in held-out samples (the prior fools it completely at log 1/2: trust one half).

Both sides are the V-way tree's predicted mean at the full code (the latents' reconstruction after all --depths
tokens, dvae/vreplay.py's VTree), so the discriminator sees nothing but what the tokens chose: real = val images
coded by the tree; fake = codes sampled from the prior a token at a time (VTree.sample) and replayed the same way.
The prior is a train_remprior.py or train_vprior.py run's prior.pt, or "uniform" (every token equally likely).

The discriminator is a small convolutional net on the latents' C x H x W picture. --n real and --n fake codes are
split into training and held-out parts (--held of each); it trains until its held-out loss has not improved for
--patience epochs and keeps its best epoch. Scores, on the held-out part: its log trust in the fakes (mean log D),
-E[log trust] beside ln 2, its loss beside 2 ln 2, and its accuracy.

What the run keeps (artifacts/<job>/): run.log, records.json, run.png (the discriminator's training and held-out
loss and log trust by epoch, and real and sampled codes decoded by the outer decoder), disc.pt.

Run from the repo: python judge_prior.py --prior artifacts/<run>/prior.pt   (or --prior uniform --tree <run>)
"""

import argparse
import copy
import json
import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dvae.artifacts import artifact  # noqa: E402
from dvae.data import get_device  # noqa: E402
from dvae.logs import RunLog  # noqa: E402
from dvae.plots import COMPONENT_COLORS, INK, MUTED, _rows_grid, _style  # noqa: E402
from dvae.remprior import RemPrior  # noqa: E402
from dvae.vprior import CandStore, CodeQA, make_net  # noqa: E402
from dvae.vreplay import VTree  # noqa: E402

LN2 = math.log(2)


class LatentDisc(nn.Module):
    """Real-or-sampled logit [B] for latents [B, C, H, W]: 3x3 convolutions, two halvings, a pooled linear head."""

    def __init__(self, C, channels=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, 2 * channels, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(2 * channels, 2 * channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(2 * channels, 4 * channels, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(4 * channels, 1))

    def forward(self, z):
        return self.net(z)[:, 0]


def load_prior(path, device):
    """A saved prior (train_remprior.py's RemPrior or a train_vprior.py network) and its run's arguments."""
    saved = torch.load(path, map_location=device, weights_only=False)
    config = dict(saved["config"])
    if config.get("kind") == "rem_prior":
        config.pop("kind")
        prior = RemPrior(**config)
    else:
        prior = make_net(config)
    missing, unexpected = prior.load_state_dict(saved["state_dict"], strict=False)
    # (RemPriors saved before the soft assignment lack its temperature: unused by a hard-count prior)
    assert not unexpected and set(missing) <= {"tau0", "log_tau"}, (missing, unexpected)
    return prior.to(device).eval(), saved.get("args", {})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prior", required=True, help="a prior.pt, or uniform")
    p.add_argument("--tree", default=None, help="the tree (default: the prior's run's)")
    p.add_argument("--depths", type=int, default=None, help="the codes' depths (default: the prior's run's)")
    p.add_argument("--temp", type=float, default=1.0, help="sampling temperature: the prior's logits divided by it "
                   "(below 1: its likelier tokens more often)")
    p.add_argument("--n", type=int, default=5000, help="real codes and fake codes each")
    p.add_argument("--held", type=float, default=0.2, help="the held-out share of each")
    p.add_argument("--chunk", type=int, default=500, help="codes sampled at once")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--bf16", action="store_true", help="the prior under bfloat16 autocast while sampling")
    p.add_argument("--data-root", default="data")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or os.path.dirname(artifact("run.log"))
    os.makedirs(out, exist_ok=True)
    device = get_device(args.device)
    log = RunLog(os.path.join(out, "run.log"), device)
    log(f"=== judge_prior.py {time.strftime('%Y-%m-%d')} {vars(args)}")
    torch.manual_seed(args.seed)
    began = time.time()

    prior, pargs = (None, {}) if args.prior == "uniform" else load_prior(args.prior, device)
    tree_run = args.tree or pargs.get("tree")
    assert tree_run, "--tree: a uniform prior has no run to take it from"
    depths = args.depths or pargs.get("depths")
    tree = VTree(tree_run, device, depths, fp16=pargs.get("fp16", False), node_batch=pargs.get("node_batch", 2048))
    K, V = tree.K, tree.args["V"]
    qa = CodeQA(tree, 1, args.data_root, args.seed)
    lat = qa.lat
    shape = tuple(lat["shape"])
    D = lat["val"][0].numel()
    log(f"prior {args.prior} ({type(prior).__name__ if prior is not None else 'uniform'}); tree {tree_run}, {K} "
        f"depths, V = {V}; latents {shape}")

    def amp():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16)

    # real: val images coded by the tree, their predicted mean at the full code
    Zv = lat["val"][:args.n].float().flatten(1)
    real = []
    t0 = time.time()
    for i in range(0, len(Zv), args.chunk):
        X = Zv[i:i + args.chunk].to(device)
        g, _ = tree.code(X, K)
        real.append((X - g.R).cpu())
        log.detail(f"    real: coded {min(i + args.chunk, len(Zv))} of {len(Zv)}")
    real = torch.cat(real)
    log(f"real: {len(real)} val images coded to depth {K} in {time.time() - t0:.0f}s")

    # fake: codes sampled from the prior
    gen = torch.Generator(device).manual_seed(args.seed + 1)
    fake = []
    t0 = time.time()
    while sum(len(f) for f in fake) < args.n:
        n = min(args.chunk, args.n - sum(len(f) for f in fake))
        def choose_uniform(d, cands, shares, parts, n=n):
            return torch.randint(0, V, (n,), device=device, generator=gen)
        choose = choose_uniform
        if prior is not None:
            store = CandStore(n, K, V, D, device, tree.mean)
            rows = torch.arange(n, device=device)
            put = store.writer(rows)
            window = getattr(prior, "window")

            def choose_prior(d, cands, shares, parts, store=store, rows=rows, put=put, n=n, window=window):
                dd = torch.full((n,), d, dtype=torch.long, device=device)
                with torch.no_grad(), amp():
                    logits = prior(store.inputs(rows, dd, window, cands.float(), parts.float())).float()
                token = torch.multinomial((logits / args.temp).softmax(-1), 1, generator=gen)[:, 0]
                put(d, cands, token, shares, parts)
                return token
            choose = choose_prior
        _, z = tree.sample(n, choose, K)
        fake.append(z.cpu())
        log.detail(f"    fake: sampled {sum(len(f) for f in fake)} of {args.n}; {time.time() - t0:.0f}s")
    fake = torch.cat(fake)
    log(f"fake: {len(fake)} codes sampled and replayed in {time.time() - t0:.0f}s; latent rms real "
        f"{real.square().mean().sqrt():.4f}, fake {fake.square().mean().sqrt():.4f}")

    # the discriminator
    C, H, W = shape
    split = int(round(len(real) * (1 - args.held)))
    perm_r = torch.randperm(len(real), generator=torch.Generator().manual_seed(args.seed + 2))
    perm_f = torch.randperm(len(fake), generator=torch.Generator().manual_seed(args.seed + 3))
    tr_x = torch.cat([real[perm_r[:split]], fake[perm_f[:split]]]).view(-1, C, H, W).to(device)
    tr_y = torch.cat([torch.ones(split), torch.zeros(split)]).to(device)
    ho_r = real[perm_r[split:]].view(-1, C, H, W).to(device)
    ho_f = fake[perm_f[split:]].view(-1, C, H, W).to(device)
    disc = LatentDisc(C, args.channels, args.dropout).to(device)
    opt = torch.optim.AdamW(disc.parameters(), lr=args.lr, weight_decay=0.01)
    log(f"discriminator: {sum(q.numel() for q in disc.parameters()) / 1e6:.2f}M parameters; {2 * split} training "
        f"codes, {len(ho_r)} real and {len(ho_f)} fake held out")

    @torch.no_grad()
    def held_out():
        disc.eval()
        sr, sf = disc(ho_r), disc(ho_f)
        loss = (F.softplus(-sr).mean() + F.softplus(sf).mean()).item()
        trust = F.logsigmoid(sf).mean().item()  # (log D on the fakes)
        acc = 0.5 * ((sr > 0).float().mean() + (sf < 0).float().mean()).item()
        return dict(loss=loss, log_trust=trust, real_log_trust=F.logsigmoid(sr).mean().item(), acc=acc)

    history = []
    best = (float("inf"), 0, copy.deepcopy(disc.state_dict()), held_out())
    for e in range(1, args.max_epochs + 1):
        disc.train()
        order = torch.randperm(len(tr_x), device=device)
        tot = 0.0
        for s in range(0, len(order), args.batch):
            j = order[s:s + args.batch]
            loss = F.binary_cross_entropy_with_logits(disc(tr_x[j]), tr_y[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(j)
        h = held_out()
        history.append(dict(epoch=e, train=2 * tot / len(order), **h))
        if h["loss"] < best[0]:
            best = (h["loss"], e, copy.deepcopy(disc.state_dict()), h)
        log(f"epoch {e}: held-out loss {h['loss']:.4f} (2 ln 2 = {2 * LN2:.4f}), log trust in fakes "
            f"{h['log_trust']:.4f} (-ln 2 = {-LN2:.4f}), accuracy {h['acc'] * 100:.1f}%; train loss "
            f"{2 * tot / len(order):.4f}; best epoch {best[1]}")
        if e - best[1] >= args.patience:
            break
    disc.load_state_dict(best[2])
    score = best[3]
    log(f"score (the discriminator of epoch {best[1]}, held out): log trust in the prior's samples "
        f"{score['log_trust']:.4f}, -E[log trust] {-score['log_trust']:.4f} (ln 2 = {LN2:.4f}: can't tell); loss "
        f"{score['loss']:.4f} (2 ln 2 = {2 * LN2:.4f}); accuracy {score['acc'] * 100:.1f}%")
    torch.save(dict(state_dict=disc.state_dict(), epoch=best[1], args=vars(args)), os.path.join(out, "disc.pt"))
    with open(os.path.join(out, "records.json"), "w") as f:
        json.dump(dict(args=vars(args), prior_args=pargs, K=K, V=V, history=history, best_epoch=best[1],
                       score=score), f)

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 2)
    ax_l, ax_t = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    ep = [h["epoch"] for h in history]
    ax_l.plot(ep, [h["loss"] for h in history], color=COMPONENT_COLORS["val"], linewidth=1.6, label="held out")
    ax_l.plot(ep, [h["train"] for h in history], color=COMPONENT_COLORS["train"], linewidth=1.2, label="train")
    ax_l.axhline(2 * LN2, color=MUTED, linestyle=":", linewidth=1.2, label="2 ln 2 (chance)")
    ax_l.legend(frameon=False, labelcolor=INK, fontsize=8)
    _style(ax_l, "Discriminator loss", "epoch", "nats")
    ax_t.plot(ep, [h["log_trust"] for h in history], color=COMPONENT_COLORS["val"], linewidth=1.6,
              label="in the prior's samples")
    ax_t.plot(ep, [h["real_log_trust"] for h in history], color=COMPONENT_COLORS["test"], linewidth=1.2,
              label="in real codes")
    ax_t.axhline(-LN2, color=MUTED, linestyle=":", linewidth=1.2, label="-ln 2 (can't tell)")
    ax_t.axvline(best[1], color=INK, linewidth=0.8, linestyle="--")
    ax_t.legend(frameon=False, labelcolor=INK, fontsize=8)
    _style(ax_t, f"Held-out log trust (kept epoch {best[1]}: {score['log_trust']:.3f} in the samples)", "epoch",
           "mean log D")
    ax = fig.add_subplot(grid[1, :])
    k = 12
    rows = [qa.decode(real[:k].to(device)).numpy(), qa.decode(fake[:k].to(device)).numpy()]
    shown = _rows_grid(rows)
    ax.imshow(shown, interpolation="nearest")
    Hh = rows[0].shape[2]
    ax.set_yticks([r * (Hh + 4) + Hh / 2 for r in range(2)])
    ax.set_yticklabels([f"real (tree, {K} depths)", "prior samples"], color=INK)
    ax.set_xticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("Codes the discriminator sees, decoded", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "run.png"), dpi=110)
    plt.close(fig)
    log(f"done in {(time.time() - began) / 60:.1f} min; {out}")


if __name__ == "__main__":
    main()
