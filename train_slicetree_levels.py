"""Train a categorical decoder for a slice tree (see README): p(x | leaf), 256 grey levels per pixel, by maximum
likelihood on the real training images routed down the tree of a trained encoder and prior.

Each image's code is the encoder's and prior's walk of it (walk_model), and its leaf the node after all K
tokens. The decoder is a half-space model like the others, read at the leaf (the history before the last
half-space, queried with the last), and its head gives 256 logits per pixel; it is trained on leaves only.
Early stopping watches held-out images' bits per pixel under it. Then the test set is scored as benchmark.py
scores a code: -log p(z(x)) - log p(x | z(x)), the prior's tokens plus the decoder, in nats per image and
bits per pixel. The tree's code is a function of the image, so this is the one-code upper bound on -log p(x)
(benchmark.py's nll_top_bound with a single code).
"""

import argparse
import json
import math
import os
import time

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")  # ~6.5 GB: fail instead of swapping
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dvae.data import load_mnist  # noqa: E402
from dvae.slicetree import D, Root, SliceModel, walk_model  # noqa: E402

LANDMARK_BUFFERS = ("lm_center", "lm_diag", "lm_U", "lm_logw")


def get_device(name):
    if name:
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def empty_cache(device):
    if device.type == "mps":
        torch.mps.empty_cache()


@torch.no_grad()
def route(encoder, prior, X, walk, chunk, log, what):
    """Each centered image's code (X [n, D]), a chunk of images at a time: (first image, its oriented
    half-spaces H [b, K, D] in half precision, o [b, K], the tokens' log probabilities under the prior [b, K]),
    on the CPU."""
    t0 = time.time()
    for i in range(0, len(X), chunk):
        _, lp, h, oo = walk_model(encoder, prior, X=X[i:i + chunk], **walk)
        yield i, h.half().cpu(), oo.cpu(), lp.cpu()
        log(f"  {what}: {min(i + chunk, len(X))}/{len(X)} images routed ({time.time() - t0:.0f}s)")


def routes(path, key, n, K, chunks):
    """The codes chunks yields, cached while key matches: the half-spaces in a memory-mapped file (path.npy,
    [n, K, D] half precision: at K = 128, 55k images take 11 GB), the rest in path.pt. Returns H (backed by
    the file), o [n, K] and the tokens' log probabilities [n, K]."""
    data, meta = path + ".npy", path + ".pt"
    if os.path.exists(data) and os.path.exists(meta):
        saved = torch.load(meta)
        if saved["key"] == key:
            return torch.from_numpy(np.load(data, mmap_mode="r+")), saved["o"], saved["logp"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    H = np.lib.format.open_memmap(data + ".tmp", mode="w+", dtype=np.float16, shape=(n, K, D))
    o, logp = torch.zeros(n, K), torch.zeros(n, K)
    for i, h, oo, lp in chunks:
        H[i:i + len(h)] = h.numpy()
        o[i:i + len(h)], logp[i:i + len(h)] = oo, lp
    H.flush()
    del H
    os.replace(data + ".tmp", data)
    torch.save(dict(key=key, o=o, logp=logp), meta)
    return torch.from_numpy(np.load(data, mmap_mode="r+")), o, logp


def leaf_nll(decoder, H, o, target, device):
    """-log p(x | leaf) in nats for a batch of leaves (H [b, K, D], o [b, K]) and their images' levels [b, D]."""
    H, o = H.to(device).float(), o.to(device)
    f = decoder.features(decoder.memory(H[:, :-1], o[:, :-1]), H[:, -1:], o[:, -1:])
    logits = decoder.levels(f)[:, 0]
    return F.cross_entropy(logits.flatten(0, 1), target.to(device).flatten(), reduction="none").view(len(H), -1).sum(1)


@torch.no_grad()
def evaluate(decoder, H, o, target, device, batch):
    """-log p(x | leaf) in nats for every image [n]."""
    decoder.eval()
    return torch.cat([leaf_nll(decoder, H[i:i + batch], o[i:i + batch], target[i:i + batch], device).cpu()
                      for i in range(0, len(H), batch)])


def bpd(nats):
    return nats / (D * math.log(2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="checkpoints/slicetree_r21.pt", help="the slice tree: its encoder and prior")
    p.add_argument("--out", default=None, help="checkpoint to write (default: --ckpt with _levels)")
    p.add_argument("--routes", default="checkpoints/slicetree_routes", help="directory caching each split's codes")
    p.add_argument("--val-samples", type=int, default=1000, help="held-out (val) images early stopping watches")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--route-chunk", type=int, default=1024, help="images walked at once")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-test", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="data")
    p.add_argument("--results", default="results/slicetree_benchmark")
    p.add_argument("--log", default="outputs/slicetree_levels.log")
    args = p.parse_args()

    device = get_device(args.device)
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    logf = open(args.log, "a")

    def log(msg):
        msg = f"{time.strftime('%H:%M:%S')} {msg}"
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"=== {time.strftime('%Y-%m-%d')} {vars(args)}")
    torch.manual_seed(args.seed)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    nets = {}
    for name in ("encoder", "prior"):
        nets[name] = SliceModel(**ckpt["configs"][name]).to(device)
        nets[name].load_state_dict(ckpt[name])
        nets[name].eval()
    encoder, prior = nets["encoder"], nets["prior"]
    root = Root(**ckpt["root"])
    a = ckpt["args"]
    walk = dict(K=a["K"], starts=a["starts"], start_depths=a["start_depths"], iters=a["iters"])

    raw = {"train": load_mnist("train", args.data_root)[:args.limit_train],
           "val": load_mnist("val", args.data_root)[:args.val_samples],
           "test": load_mnist("test", args.data_root)[:args.limit_test]}
    H, o, logp, target = {}, {}, {}, {}
    for split, images in raw.items():
        X = ((images.float() + 0.5) / 256).flatten(1).to(device) - root.center  # each grey level's bin center
        key = dict(ckpt=os.path.abspath(args.ckpt), split=split, n=len(images), version=2)
        path = os.path.join(args.routes, f"{os.path.splitext(os.path.basename(args.ckpt))[0]}_{split}")
        H[split], o[split], logp[split] = routes(path, key, len(images), walk["K"],
                                                 route(encoder, prior, X, walk, args.route_chunk, log, split))
        target[split] = images.long().flatten(1)
        empty_cache(device)
        log(f"{split}: {len(images)} codes, tokens {bpd(-logp[split].sum(1)).mean().item():.4f} bits/dim")

    cfg = dict(ckpt["configs"]["encoder"], outputs=("levels",))
    decoder = SliceModel(**cfg).to(device)
    with torch.no_grad():
        for name in LANDMARK_BUFFERS:
            getattr(decoder, name).copy_(getattr(encoder, name))
    log(f"categorical decoder: {sum(q.numel() for q in decoder.parameters()) / 1e6:.1f}M parameters")
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=0.01, fused=device.type in ("mps", "cuda"))
    out = args.out or args.ckpt.replace(".pt", "_levels.pt")

    def save():
        torch.save(dict(ckpt, levels=decoder.state_dict(), configs=dict(ckpt["configs"], levels=decoder.config)), out)

    best: tuple[float, int, dict | None] = (math.inf, -1, None)
    N = len(H["train"])
    for epoch in range(args.epochs):
        t0 = time.time()
        decoder.train()
        total, count = 0.0, 0
        order = torch.randperm(N)
        for step, i in enumerate(range(0, N, args.batch)):
            sel = order[i:i + args.batch]
            nll = leaf_nll(decoder, H["train"][sel], o["train"][sel], target["train"][sel], device)
            loss = bpd(nll).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            opt.step()
            total, count = total + loss.item() * len(sel), count + len(sel)
            if step % 200 == 199:
                empty_cache(device)
        t1 = time.time()
        val = bpd(evaluate(decoder, H["val"], o["val"], target["val"], device, args.batch)).mean().item()
        log(f"  epoch {epoch:2d}: train {total / count:.4f}, held-out {val:.4f} bits/dim at the leaf "
            f"({t1 - t0:.0f}s train, {time.time() - t1:.0f}s val)")
        if val < best[0]:
            best = (val, epoch, {key: v.detach().clone() for key, v in decoder.state_dict().items()})
            save()
        elif epoch - best[1] >= args.patience:
            break
    assert best[2] is not None
    decoder.load_state_dict(best[2])
    log(f"  kept epoch {best[1]} ({best[0]:.4f} held-out bits/dim)")

    rows = {}
    for split in ("val", "test"):
        tokens = -logp[split].sum(1)
        leaf = evaluate(decoder, H[split], o[split], target[split], device, args.batch)
        nll = tokens + leaf
        rows[split] = dict(images=len(nll), nll_nats=nll.mean().item(), nll_bpd=bpd(nll).mean().item(),
                           tokens_nats=tokens.mean().item(), tokens_bpd=bpd(tokens).mean().item(),
                           leaf_nats=leaf.mean().item(), leaf_bpd=bpd(leaf).mean().item())
        log(f"{split}: {rows[split]['nll_nats']:.2f} nats/image, {rows[split]['nll_bpd']:.4f} bits/dim "
            f"(tokens {rows[split]['tokens_bpd']:.4f}, leaf {rows[split]['leaf_bpd']:.4f}) over {len(nll)} images")
    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
    with open(args.results + ".json", "w") as f:
        json.dump(dict(ckpt=args.ckpt, decoder=out, bound="one code: -log p(z(x)) - log p(x | z(x))", **rows), f, indent=2)
    with open(args.results + ".md", "w") as f:
        f.write("| split | NLL bound (nats/img) | NLL bound (bpd) | tokens (bpd) | leaf (bpd) |\n|---|---|---|---|---|\n")
        for split, r in rows.items():
            f.write(f"| {split} | {r['nll_nats']:.2f} | {r['nll_bpd']:.4f} | {r['tokens_bpd']:.4f} | {r['leaf_bpd']:.4f} |\n")
    save()
    log(f"wrote {out}, {args.results}.json and .md")


if __name__ == "__main__":
    main()
