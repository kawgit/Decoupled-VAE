"""A remainder-model prior over a V-way tree's codes (dvae/remprior.py's RemPrior): at each node, a convolutional net
reads the current mean and the last delta and predicts the remainder's Gaussian; a token's probability is the share
of remainders drawn from it that fall in the token's cell (the nearest of the partition centroids).

The codes come from replaying the tree's saved networks (dvae/vreplay.py) into CandStores on the GPU, as in
train_vprior.py: a rolling buffer of --buffer training images (--rotate replaced each epoch, scored first:
"incoming"), and --val and --test images. The net trains on the remainders' negative log likelihood (nats per latent
number), every (image, depth) pair a sample. Every epoch logs val's likelihood beside the isotropic baseline (a
Gaussian of each depth's rms at the node's mean: the untrained net), and the bits per image of --bits-images val
images' codes under the prior, beside the same baseline's and uniform's. Early stopping keeps the epoch of the
lowest val likelihood; the kept prior's bits are then scored on all of val and test.

run.png (redrawn every epoch): likelihood and bits by epoch, the kept prior's bits saved per token by depth, and QA
images (dvae/vprior.py's CodeQA: test images, the tree's reconstructions, codes sampled from the prior and from the
tree's shares, decoded).

What the run keeps (artifacts/<job>/): run.log (and run.full.log), records.json, run.png, prior.pt.

Run from the repo: python train_remprior.py --tree artifacts/<inner job> --depths 512
"""

import argparse
import copy
import json
import math
import os
import time

import torch

from dvae.artifacts import artifact
from dvae.data import get_device
from dvae.logs import RunLog
from dvae.plots import plot_remprior
from dvae.remprior import RemPrior
from dvae.vprior import CandStore, CodeQA
from dvae.vreplay import VTree

LN2 = math.log(2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tree", required=True, metavar="RUN")
    p.add_argument("--depths", type=int, default=None)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--rank", type=int, default=0, help="shared directions of the remainder's covariance")
    p.add_argument("--samples", type=int, default=256, help="remainders drawn per node for token probabilities")
    p.add_argument("--batch", type=int, default=256, help="(image, depth) pairs")
    p.add_argument("--steps", type=int, default=1000, help="batches per epoch")
    p.add_argument("--buffer", type=int, default=4000)
    p.add_argument("--rotate", type=int, default=400, help="buffer images replaced each epoch (0: a fixed buffer)")
    p.add_argument("--val", type=int, default=1000)
    p.add_argument("--test", type=int, default=1000)
    p.add_argument("--bits-images", type=int, default=100, help="val images whose bits are scored each epoch")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--progress-every", type=int, default=200)
    p.add_argument("--chunk", type=int, default=1000, help="images replayed at once")
    p.add_argument("--node-batch", type=int, default=2048)
    p.add_argument("--fp16", action="store_true", help="the tree's replay under float16 (V100s)")
    p.add_argument("--bf16", action="store_true", help="the net under bfloat16 autocast")
    p.add_argument("--qa-n", type=int, default=10)
    p.add_argument("--qa-every", type=int, default=5)
    p.add_argument("--data-root", default="data")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or os.path.dirname(artifact("run.log"))
    os.makedirs(out, exist_ok=True)
    device = get_device(args.device)
    log = RunLog(os.path.join(out, "run.log"), device)
    log(f"=== train_remprior.py {time.strftime('%Y-%m-%d')} {vars(args)}")
    torch.manual_seed(args.seed)
    began = time.time()

    def amp():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16)

    tree = VTree(args.tree, device, args.depths, fp16=args.fp16, node_batch=args.node_batch)
    K, V = tree.K, tree.args["V"]
    qa = CodeQA(tree, args.qa_n, args.data_root, args.seed)
    lat = qa.lat
    shape = tuple(lat["shape"])
    D = lat["train"][0].numel()
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
            Zi = Z[i:i + args.chunk].to(device).float()
            tree.code(Zi, K, visit=store.writer(rows[i:i + args.chunk], Zi))
            done = min(i + args.chunk, len(rows))
            log.detail(f"    {label}: replayed {done} of {len(rows)} images; "
                       f"{(len(rows) - done) * (time.time() - t0) / done:.0f}s left")
        return time.time() - t0

    stores = {}
    for name, n, Z in (("buffer", args.buffer, None), ("val", args.val, lat["val"]), ("test", args.test, lat["test"])):
        stores[name] = CandStore(n, K, V, D, device, tree.mean, parts=True, images=True)
        Z = pool[take(n)] if Z is None else Z[:n].flatten(1)
        seconds = fill(stores[name], torch.arange(n, device=device), Z, name)
        log(f"{name}: {n} images replayed in {seconds:.0f}s ({stores[name].nbytes() / 2**30:.1f} GiB on the GPU)")
    buf = stores["buffer"]

    def remainders(store, i, d):
        return store.x[i].float() - store.base - store.cum[i, d].float()

    with torch.no_grad():
        rms = torch.stack([remainders(buf, torch.arange(buf.n, device=device), torch.full((buf.n,), d, device=device))
                           .square().mean().sqrt() for d in range(K)])
    prior = RemPrior(K, V, shape, args.channels, args.blocks, args.rank, args.samples).to(device)
    prior.set_scales(rms)
    size = sum(q.numel() for q in prior.parameters()) / 1e6
    log(f"tree {args.tree}: {K} depths, V = {V}, latents {shape}; rms remainder per number {rms[0]:.4g} at depth 0, "
        f"{rms[K // 2]:.4g} at {K // 2}, {rms[-1]:.4g} at {K - 1}; prior {size:.2f}M parameters, rank {args.rank}, "
        f"{args.samples} samples per node")

    def pairs(n, per):
        """Every (image, depth) of n images, in batches of about per pairs: (i, d) tensors."""
        imgs = max(1, per // K)
        for s in range(0, n, imgs):
            r = torch.arange(s, min(s + imgs, n), device=device)
            yield r.repeat_interleave(K), torch.arange(K, device=device).repeat(len(r))

    @torch.no_grad()
    def evaluate_nll(store, n):
        """Val likelihood: the prior's and the isotropic baseline's, nats per latent number."""
        prior.eval()
        tot = base = 0.0
        count = 0
        for i, d in pairs(n, 4096):
            x = store.inputs(i, d, 1)
            R = remainders(store, i, d)
            with amp():
                tot += prior.nll(x, R).sum().item()
            s = prior.scale[d][:, None]
            base += (0.5 * ((2 * math.pi * s.square()).log() + R.square() / s.square())).mean(-1).sum().item()
            count += len(i)
        return tot / count, base / count

    @torch.no_grad()
    def evaluate_bits(store, n, net):
        """Bits per image of n images' codes under net (by depth too)."""
        net.eval()
        bits = torch.zeros(K, device=device)
        for i, d in pairs(n, 2048):
            with amp():
                lp = net.token_logprobs(store.inputs(i, d, 1))
            bits.index_add_(0, d, -lp.gather(-1, store.tok[i, d].long()[:, None])[:, 0] / LN2)
        by_depth = (bits / n).tolist()
        return sum(by_depth), by_depth

    iso = copy.deepcopy(prior)  # (the untrained net: the isotropic baseline)
    nb = min(args.bits_images, args.val)
    iso_bits, iso_depth = evaluate_bits(stores["val"], nb, iso)
    log(f"isotropic baseline: {iso_bits:.2f} bits per image on {nb} val images (uniform {K * math.log2(V):.0f})")

    records: dict = dict(args=vars(args), K=K, V=V, history=[], baseline=dict(bits=iso_bits, by_depth=iso_depth),
                         final=None, by_depth=None)
    samples: dict = {}

    def save():
        atomic_json(records, os.path.join(out, "records.json"))
        plot_remprior(records, os.path.join(out, "run.png"), samples or None)

    opt = torch.optim.AdamW(prior.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = (float("inf"), 0, copy.deepcopy(prior.state_dict()))
    oldest = 0
    for e in range(1, args.max_epochs + 1):
        t0 = time.time()
        incoming = None
        if args.rotate and e > 1:
            rows = (oldest + torch.arange(args.rotate, device=device)) % args.buffer
            oldest = (oldest + args.rotate) % args.buffer
            fill(buf, rows, pool[take(args.rotate)], "rotation")
            prior.eval()
            with torch.no_grad():
                tot = 0.0
                for s in range(0, len(rows), 8):
                    r = rows[s:s + 8].repeat_interleave(K)
                    dd = torch.arange(K, device=device).repeat(len(rows[s:s + 8]))
                    with amp():
                        tot += prior.nll(buf.inputs(r, dd, 1), remainders(buf, r, dd)).sum().item()
                incoming = tot / (len(rows) * K)
        prior.train()
        t1 = time.time()
        total = 0.0
        for b in range(args.steps):
            i = torch.randint(0, args.buffer, (args.batch,), device=device)
            d = torch.randint(0, K, (args.batch,), device=device)
            with amp():
                loss = prior.nll(buf.inputs(i, d, 1), remainders(buf, i, d)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
            opt.step()
            total += loss.item()
            if (b + 1) % args.progress_every == 0:
                rate = (b + 1) / (time.time() - t1)
                log.detail(f"    epoch {e}: batch {b + 1} of {args.steps}; train nll {total / (b + 1):.5f} nats per "
                           f"number; {rate:.1f} batches/s, {(args.steps - b - 1) / rate:.0f}s left")
        train_s = time.time() - t1
        val, val_iso = evaluate_nll(stores["val"], args.val)
        bits, by_depth = evaluate_bits(stores["val"], nb, prior)
        if val < best[0]:
            best = (val, e, copy.deepcopy(prior.state_dict()))
            records["by_depth"] = dict(epoch=e, prior=by_depth, baseline=iso_depth)
        if e == 1 or e % args.qa_every == 0:
            samples.update(qa.samples(prior, amp), epoch=e)
        records["history"].append(dict(epoch=e, train=total / args.steps, val=val, val_iso=val_iso, incoming=incoming,
                                       bits=bits, seconds=time.time() - t0, train_seconds=train_s,
                                       hours=(time.time() - began) / 3600))
        seen = f"incoming {incoming:.5f}, " if incoming is not None else ""
        log(f"epoch {e}: val nll {val:.5f} nats per number (isotropic {val_iso:.5f}; {seen}train "
            f"{total / args.steps:.5f}); bits per image on {nb} val images {bits:.2f} (isotropic {iso_bits:.2f}, "
            f"uniform {K * math.log2(V):.0f}); best epoch {best[1]}; {time.time() - t0:.0f}s (train {train_s:.0f})")
        save()
        if e - best[1] >= args.patience:
            break

    prior.load_state_dict(best[2])
    torch.save(dict(config=prior.config, state_dict=prior.state_dict(), epoch=best[1], args=vars(args)),
               os.path.join(out, "prior.pt"))
    samples.update(qa.samples(prior, amp), epoch=best[1])
    final = {}
    for s in ("val", "test"):
        b, _ = evaluate_bits(stores[s], stores[s].n, prior)
        bi, _ = evaluate_bits(stores[s], stores[s].n, iso)
        final[s] = dict(bits=b, isotropic=bi)
    records["final"] = final
    log(f"kept the prior of epoch {best[1]} (val nll {best[0]:.5f}): bits per image val {final['val']['bits']:.2f} "
        f"(isotropic {final['val']['isotropic']:.2f}), test {final['test']['bits']:.2f} (isotropic "
        f"{final['test']['isotropic']:.2f}); uniform {K * math.log2(V):.0f}")
    save()
    log(f"done in {(time.time() - began) / 60:.1f} min; {out}")


def atomic_json(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
