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
from dvae.data import get_device, load_images
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
    p.add_argument("--history", type=int, default=4, help="the last deltas read beside the current mean")
    p.add_argument("--token-weight", type=float, default=0.0, help="> 0: soft token assignment (dvae/remprior.py), "
                   "the loss adds this times the tokens' cross-entropy (nats per token), and early stopping and the "
                   "learning rate follow the scored val images' bits instead of the likelihood")
    p.add_argument("--token-samples", type=int, default=16, help="remainders drawn per pair for the token loss")
    p.add_argument("--calib-images", type=int, default=1000,
                   help="images not trained on whose tokens fit the calibration (a temperature and bias per depth)")
    p.add_argument("--calib-steps", type=int, default=500)
    p.add_argument("--final-samples", type=int, default=1024, help="remainders drawn per node for the final scores")
    p.add_argument("--samples", type=int, default=256, help="remainders drawn per node for token probabilities")
    p.add_argument("--batch", type=int, default=256, help="(image, depth) pairs")
    p.add_argument("--steps", type=int, default=1000, help="batches per epoch")
    p.add_argument("--buffer", type=int, default=4000)
    p.add_argument("--rotate", type=int, default=400, help="buffer images replaced each epoch (0: a fixed buffer)")
    p.add_argument("--val", type=int, default=1000)
    p.add_argument("--test", type=int, default=1000)
    p.add_argument("--bits-images", type=int, default=100, help="val images whose bits are scored each epoch")
    p.add_argument("--flip", action="store_true", help="the training pool also holds the latents of the training "
                   "images mirrored left to right (encoded by the outer autoencoder: real latents of real images)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=6, help="epochs without a lower val likelihood (the learning "
                   "rate halves after each 2)")
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--max-minutes", type=float, default=None, help="stop training once this many minutes have "
                   "passed since the start (then calibrate, score and save: a run that must end before its slot does)")
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
    if args.flip:
        pool = torch.cat([pool, flipped_latents(qa, lat, args.data_root, device, log)])
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
    prior = RemPrior(K, V, shape, args.channels, args.blocks, args.rank, args.samples, args.history,
                     soft=args.token_weight > 0).to(device)
    W = args.history
    prior.set_scales(rms)
    with torch.no_grad():  # (each depth's gap between the nearest and second nearest centroid, squared distances)
        gaps = []
        for d in range(K):
            r = torch.arange(buf.n, device=device)
            dd = torch.full((buf.n,), d, device=device)
            q = buf.part[r, d].float() * buf.pscale[r, d][:, None, None] if buf.part is not None else None
            assert q is not None and buf.pscale is not None
            d2 = (q.square().sum(-1) - 2 * torch.einsum("nd,nvd->nv", remainders(buf, r, dd), q)).sort(-1).values
            gaps.append((d2[:, 1] - d2[:, 0]).mean())
        prior.set_taus(torch.stack(gaps))
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
            x = store.inputs(i, d, W)
            R = remainders(store, i, d)
            with amp():
                tot += prior.nll(x, R).sum().item()
            s = prior.scale[d][:, None]
            base += (0.5 * ((2 * math.pi * s.square()).log() + R.square() / s.square())).mean(-1).sum().item()
            count += len(i)
        return tot / count, base / count

    @torch.no_grad()
    def evaluate_bits(store, n, net, calibrated=True, samples=None):
        """Bits per image of n images' codes under net (by depth too)."""
        net.eval()
        bits = torch.zeros(K, device=device)
        for i, d in pairs(n, 2048):
            with amp():
                lp = raw_logprobs(net, store.inputs(i, d, W), samples)
            if calibrated:
                lp = net.calibrate(lp, d)
            bits.index_add_(0, d, -lp.gather(-1, store.tok[i, d].long()[:, None])[:, 0] / LN2)
        by_depth = (bits / n).tolist()
        return sum(by_depth), by_depth

    iso = copy.deepcopy(prior)  # (the untrained net: the isotropic baseline, hard counts)
    iso.soft = False
    nb = min(args.bits_images, args.val)
    iso_bits, iso_depth = evaluate_bits(stores["val"], nb, iso)
    log(f"isotropic baseline: {iso_bits:.2f} bits per image on {nb} val images (uniform {K * math.log2(V):.0f})")

    records: dict = dict(args=vars(args), K=K, V=V, history=[], baseline=dict(bits=iso_bits, by_depth=iso_depth),
                         final=None, by_depth=None)
    samples: dict = {}

    def save():
        atomic_json(records, os.path.join(out, "records.json"))
        plot_remprior(records, os.path.join(out, "run.png"), samples or None)

    model_params = [q for name, q in prior.named_parameters() if name not in ("temp", "bias")]
    opt = torch.optim.AdamW(model_params, lr=args.lr, weight_decay=args.weight_decay)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
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
                        tot += prior.nll(buf.inputs(r, dd, W), remainders(buf, r, dd)).sum().item()
                incoming = tot / (len(rows) * K)
        prior.train()
        t1 = time.time()
        total = total_ce = 0.0
        for b in range(args.steps):
            i = torch.randint(0, args.buffer, (args.batch,), device=device)
            d = torch.randint(0, K, (args.batch,), device=device)
            with amp():
                x = buf.inputs(i, d, W)
                loss = prior.nll(x, remainders(buf, i, d)).mean()
                if args.token_weight > 0:
                    ce = -prior.token_logprobs_soft(x, args.token_samples).gather(-1, buf.tok[i, d].long()[:, None])
                    loss = loss + args.token_weight * ce.mean()
                    total_ce += ce.mean().item()
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
        metric = bits if args.token_weight > 0 else val
        plateau.step(metric)
        if metric < best[0]:
            best = (metric, e, copy.deepcopy(prior.state_dict()))
            records["by_depth"] = dict(epoch=e, prior=by_depth, baseline=iso_depth)
        if e == 1 or e % args.qa_every == 0:
            samples.update(qa.samples(prior, amp), epoch=e)
        records["history"].append(dict(epoch=e, train=total / args.steps, train_ce=total_ce / args.steps, val=val,
                                       val_iso=val_iso, incoming=incoming, bits=bits, seconds=time.time() - t0,
                                       train_seconds=train_s, hours=(time.time() - began) / 3600))
        seen = f"incoming {incoming:.5f}, " if incoming is not None else ""
        log(f"epoch {e}: val nll {val:.5f} nats per number (isotropic {val_iso:.5f}; {seen}train "
            f"{total / args.steps:.5f}" + (f", train token CE {total_ce / args.steps * K / LN2:.2f} bits per image"
                                           if args.token_weight > 0 else "")
            + f"); bits per image on {nb} val images {bits:.2f} (isotropic {iso_bits:.2f}, "
            f"uniform {K * math.log2(V):.0f}); best epoch {best[1]}; lr {opt.param_groups[0]['lr']:.2g}; "
            f"{time.time() - t0:.0f}s (train {train_s:.0f})")
        save()
        if e - best[1] >= args.patience:
            break
        if args.max_minutes is not None and time.time() - began > 60 * args.max_minutes:
            log(f"stopping: {args.max_minutes:g} minutes have passed")
            break

    prior.load_state_dict(best[2])
    # (calibration: on images never trained on, the remainder model frozen and its sampled probabilities computed
    # once; temp and bias by cross-entropy)
    cal = CandStore(args.calib_images, K, V, D, device, tree.mean, parts=True, images=True)
    fill(cal, torch.arange(args.calib_images, device=device), pool[take(args.calib_images)], "calibration")
    raws, toks, ds = [], [], []
    with torch.no_grad():
        for i, d in pairs(args.calib_images, 2048):
            with amp():
                raws.append(raw_logprobs(prior, cal.inputs(i, d, W)))
            toks.append(cal.tok[i, d].long())
            ds.append(d)
    raw, tok, dd = torch.cat(raws), torch.cat(toks), torch.cat(ds)
    del cal
    copt = torch.optim.Adam([prior.temp, prior.bias], lr=0.02)
    before = -prior.calibrate(raw, dd).gather(-1, tok[:, None]).mean().item()
    for _ in range(args.calib_steps):
        loss = -prior.calibrate(raw, dd).gather(-1, tok[:, None]).mean()
        copt.zero_grad(set_to_none=True)
        loss.backward()
        copt.step()
    log(f"calibrated on {args.calib_images} unseen images' tokens: {before * K / LN2:.2f} -> "
        f"{loss.item() * K / LN2:.2f} bits per image; temperature by depth {prior.temp[:K].min().item():.2f} .. "
        f"{prior.temp[:K].max().item():.2f}")
    torch.save(dict(config=prior.config, state_dict=prior.state_dict(), epoch=best[1], args=vars(args)),
               os.path.join(out, "prior.pt"))
    samples.update(qa.samples(prior, amp), epoch=best[1])
    final = {}
    for s in ("val", "test"):
        b, b_depth = evaluate_bits(stores[s], stores[s].n, prior, samples=args.final_samples)
        braw, _ = evaluate_bits(stores[s], stores[s].n, prior, calibrated=False, samples=args.final_samples)
        bi, _ = evaluate_bits(stores[s], stores[s].n, iso, calibrated=False, samples=args.final_samples)
        final[s] = dict(bits=b, uncalibrated=braw, isotropic=bi)
        if s == "val":
            records["by_depth"] = dict(epoch=best[1], prior=b_depth, baseline=records["baseline"]["by_depth"])
    records["final"] = final
    log(f"kept the prior of epoch {best[1]} (val {'bits' if args.token_weight > 0 else 'nll'} {best[0]:.5f}), "
        f"{args.final_samples} samples per node: bits per "
        f"image val {final['val']['bits']:.2f} (uncalibrated {final['val']['uncalibrated']:.2f}, isotropic "
        f"{final['val']['isotropic']:.2f}), test {final['test']['bits']:.2f} (uncalibrated "
        f"{final['test']['uncalibrated']:.2f}, isotropic {final['test']['isotropic']:.2f}); uniform "
        f"{K * math.log2(V):.0f}")
    save()
    log(f"done in {(time.time() - began) / 60:.1f} min; {out}")


@torch.no_grad()
def flipped_latents(qa: CodeQA, lat: dict, data_root: str, device, log, batch: int = 1000) -> torch.Tensor:
    """The training images mirrored left to right, encoded and scaled as train_outer.py made latents.pt
    (float16, flattened); first checked by re-encoding the unmirrored ones against the stored latents."""
    images = load_images(lat["dataset"], "train", data_root)[:len(lat["train"])]
    C, h, w = lat["shape"]
    mean, std = lat["mean"].to(device)[None, :, None, None], lat["std"].to(device)[None, :, None, None]
    qa.model.eval()

    def encode(x):
        z = qa.model.encoder(((x.to(device).float() + 0.5) / 256).view(len(x), *images.shape[1:])).float()
        return ((z - mean) / std * lat["scale"]).flatten(1)

    check = encode(images[:batch])
    stored = lat["train"][:batch].float().flatten(1).to(device)
    log(f"re-encoded {batch} training images: mse to the stored latents {(check - stored).square().mean():.3g} "
        f"(their variance {stored.var():.3g})")
    out = [encode(images[i:i + batch].flip(-1)).half().cpu() for i in range(0, len(images), batch)]
    flipped = torch.cat(out)
    log(f"{len(flipped)} mirrored training images encoded; mean square distance to their unmirrored latents mirrored "
        f"{(flipped[:batch].float().view(-1, C, h, w) - lat['train'][:batch].float().flip(-1)).square().mean():.3g}")
    return flipped


@torch.no_grad()
def raw_logprobs(net: RemPrior, x, samples=None):
    """net's uncalibrated token log probabilities: its soft assignment's if it has one, else hard counts."""
    if net.soft:
        return net.token_logprobs_soft(x, samples or net.samples)
    return net.token_logprobs(x, samples)


def atomic_json(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
