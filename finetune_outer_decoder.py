"""Fine-tune a train_outer.py run's decoder on a train_inner_vway.py tree's reconstructions: the decoder learns to
turn the latents the tree reconstructs after K depths (not the encoder's exact ones) into the images, correcting
what the tree's codes get wrong.

The outer run's training, val and test images (its latents.pt, and dvae/data.py's pixels for them) are coded through
the tree's first K depths (dvae/vreplay.py, the tree's saved networks; K defaults to every depth the tree has saved);
the codes are checked against the tree run's own records (its val and test mse at depth K). Then the whole decoder
trains on the training images' reconstructions against their pixels (AdamW, early-stopped on val's pixel mse,
logged each epoch), and test's pixel mse is reported before and after at depth K, at K / 2 and K / 4 (does the
correction carry over to other depths?) and on the encoder's exact latents (the outer's own floor).

What the run keeps (artifacts/<job>/): run.log, records.json, decoder_dK.pt (the fine-tuned decoder's state).

Run from the repo: python finetune_outer_decoder.py --outer artifacts/<outer job>/outer --tree artifacts/<tree job>
"""

import argparse
import copy
import json
import math
import os
import time

import torch

from dvae.artifacts import artifact
from dvae.data import get_device, load_images
from dvae.logs import RunLog
from dvae.outer import load_outer
from dvae.vreplay import VTree


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outer", required=True, metavar="RUN", help="the train_outer.py run whose latents the tree coded")
    p.add_argument("--tree", required=True, metavar="RUN", help="a train_inner_vway.py run that saved its networks")
    p.add_argument("--depth", type=int, default=None, help="K: the depths coded (default: every depth saved)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--chunk", type=int, default=20000, help="images coded at once")
    p.add_argument("--node-batch", type=int, default=2048)
    p.add_argument("--fp16", action="store_true", help="float16 autocast (V100s) for the tree and the decoder")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--limit-test", type=int, default=None)
    p.add_argument("--data-root", default="data")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or os.path.dirname(artifact("run.log"))
    os.makedirs(out, exist_ok=True)
    device = get_device(args.device)
    log = RunLog(os.path.join(out, "run.log"), device)
    log(f"=== finetune_outer_decoder.py {time.strftime('%Y-%m-%d')} {vars(args)}")
    torch.manual_seed(args.seed)
    began = time.time()

    model, lat = load_outer(args.outer, device)
    C, h, w = lat["shape"]
    mean, std = lat["mean"].to(device)[None, :, None, None], lat["std"].to(device)[None, :, None, None]
    limits = dict(train=args.limit_train, val=args.limit_val, test=args.limit_test)
    Z = {s: lat[s][:lim].float().flatten(1) for s, lim in limits.items()}
    images = {s: load_images(lat["dataset"], s, args.data_root)[:len(Z[s])] for s in Z}
    for s in Z:
        assert len(images[s]) == len(Z[s]), (s, len(images[s]), len(Z[s]))
    log(f"{', '.join(f'{len(Z[s])} {s}' for s in Z)} images; latents {C} x {h} x {w}")

    tree = VTree(args.tree, device, args.depth, fp16=args.fp16, node_batch=args.node_batch)
    K = tree.K
    marks = sorted({K, K // 2, K // 4})
    log(f"tree {args.tree}: {K} depths ({K * math.log2(tree.args['V']):.0f} bits), "
        f"{sum(len(r) for r in tree.nets)} networks; coding at depths {marks}")

    # (every image coded through the tree, a chunk at a time; reconstructions at the marks kept on the host)
    recon: dict[str, dict[int, torch.Tensor]] = {}
    mse: dict[str, float] = {}
    for s, X in Z.items():
        t0, parts, sq = time.time(), [], 0.0
        for i in range(0, len(X), args.chunk):
            g, shots = tree.code(X[i:i + args.chunk].to(device), K, marks)
            sq += g.R.square().sum().item()
            parts.append(shots)
        recon[s] = {k: torch.cat([sh[k] for sh in parts]) for k in marks}
        mse[s] = sq / X.numel()
        log(f"{s}: coded in {time.time() - t0:.0f}s; latent mse at depth {K} {mse[s]:.6f}")
    ran = tree.records["depths"][K - 1]
    log(f"check against the tree run's records at depth {K}: val {ran['val']:.6f} (here {mse['val']:.6f}), test "
        f"{ran['test']:.6f} (here {mse['test']:.6f}); train here {mse['train']:.6f} (the tree's buffer held these "
        f"images a few depths each: near test's means near unseen)")

    def decode(z, train=False):
        x = z.float().view(-1, C, h, w) / lat["scale"] * std + mean
        with torch.autocast(device.type, dtype=torch.float16 if args.fp16 else torch.bfloat16,
                            enabled=args.fp16 or args.bf16):
            y = model.decoder(x).float()
        return y if train else y.clamp(0, 1)

    def target(idx, s):
        return (images[s][idx].to(device, non_blocking=True).float() + 0.5) / 256  # (dvae.deltatree.pixels)

    @torch.no_grad()
    def px_mse(s, z, batch=250):
        model.decoder.eval()
        total = 0.0
        for i in range(0, len(z), batch):
            idx = torch.arange(i, min(i + batch, len(z)))
            total += (decode(z[idx].to(device)) - target(idx, s)).square().sum().item()
        return total / (len(z) * images[s][0].numel())

    def scores(tag):
        res = {f"test_d{k}": px_mse("test", recon["test"][k]) for k in marks}
        res["test_exact"] = px_mse("test", Z["test"])
        log(f"{tag}: test pixel mse " + ", ".join(f"depth {k} {res[f'test_d{k}']:.6f}" for k in marks)
            + f"; exact latents {res['test_exact']:.6f}")
        return res

    before = scores("before")
    v0 = px_mse("val", recon["val"][K])
    log(f"before: val pixel mse at depth {K} {v0:.6f}")
    opt = torch.optim.AdamW(model.decoder.parameters(), lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler(device.type, enabled=args.fp16)
    best = (v0, 0, copy.deepcopy(model.decoder.state_dict()))
    history = [dict(epoch=0, val=v0)]
    zt = recon["train"][K]
    for e in range(1, args.max_epochs + 1):
        t0 = time.time()
        model.decoder.train()
        total, n = 0.0, 0
        for idx in torch.randperm(len(zt)).split(args.batch):
            loss = (decode(zt[idx].to(device, non_blocking=True), train=True) - target(idx, "train")).square().mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total, n = total + loss.item() * len(idx), n + len(idx)
        v = px_mse("val", recon["val"][K])
        if v < best[0]:
            best = (v, e, copy.deepcopy(model.decoder.state_dict()))
        history.append(dict(epoch=e, train=total / n, val=v, seconds=time.time() - t0))
        log(f"epoch {e}: train pixel mse {total / n:.6f}, val {v:.6f} (before {v0:.6f}; best {best[0]:.6f} at epoch "
            f"{best[1]}); {time.time() - t0:.0f}s")
        if e - best[1] >= args.patience:
            break
    model.decoder.load_state_dict(best[2])
    after = scores(f"after (epoch {best[1]})")
    log("change on test: " + ", ".join(f"{k.removeprefix('test_')} {after[k] / before[k] - 1:+.1%}" for k in before))
    torch.save(dict(state_dict={k: v.cpu() for k, v in model.decoder.state_dict().items()}, depth=K, tree=args.tree,
                    outer=args.outer, epoch=best[1]), os.path.join(out, f"decoder_d{K}.pt"))
    with open(os.path.join(out, "records.json"), "w") as f:
        json.dump(dict(args=vars(args), depth=K, marks=marks, latent_mse=mse, tree_record=ran, before=before,
                       after=after, history=history, seconds=time.time() - began), f)
    log(f"done in {(time.time() - began) / 60:.1f} min; {out}")


if __name__ == "__main__":
    main()
