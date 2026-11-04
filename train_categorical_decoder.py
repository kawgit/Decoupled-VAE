"""Stage 4: train the categorical decoder (dvae/deltatree.py's LeafDecoder), p(x | code): 256 grey levels per
pixel given a code's first --tokens tokens, by maximum likelihood on the training images, on the codes
dump_autoencoder.py wrote.

It reads a code's delta history and its node's predicted mean image (the training mean plus the deltas), so it
adds what the autoencoder's mean leaves out: each pixel's spread around it. Every epoch logs and plots
(dvae/plots.py's plot_categorical) the val and train images' pixel bits per dimension under it, and early
stopping watches val. Then val and test are scored as -log2 p(code) - log2 p(x | code), the one-code upper bound
on -log2 p(x): the code priced by --prior (train_prior.py's) when given, else by the autoencoder's own shares.

Run from the repo: python train_categorical_decoder.py --codes artifacts/<dump job>
                   [--prior artifacts/<prior job>/prior.pt]
Next: qa_categorical_decoder.py draws images from codes the prior samples.
"""

import argparse
import json
import math
import os
import time

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")  # ~6.5 GB: fail instead of swapping
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dvae.artifacts import artifact  # noqa: E402
from dvae.codes import Codes, bits_of, deltas, load_codes, own_shares  # noqa: E402
from dvae.data import get_device, load_mnist  # noqa: E402
from dvae.deltatree import D, LeafDecoder  # noqa: E402
from dvae.logs import RunLog  # noqa: E402
from dvae.plots import plot_categorical  # noqa: E402
from dvae.prior import load_prior, token_bits  # noqa: E402


class Leaves:
    """One split's images at their codes' nodes after k tokens, as the decoder reads them: delta histories
    [N, k, D] (half precision), sides and ms [N, k], predicted means [N, D] and grey levels [N, D] (uint8), on
    the CPU."""

    def __init__(self, codes: Codes, images, k, chunk=4096):
        self.deltas = torch.cat([deltas(codes.r[i:i + chunk, :k].float(), codes.m[i:i + chunk, :k].float(),
                                        codes.rare[i:i + chunk, :k]).half() for i in range(0, len(codes), chunk)])
        self.sides = codes.rare[:, :k].float() * 2 - 1
        self.ms = codes.m[:, :k].float()
        self.mu = torch.cat([codes.mean + self.deltas[i:i + chunk].float().sum(1)
                             for i in range(0, len(codes), chunk)])
        self.levels = images.flatten(1)

    def __len__(self):
        return len(self.levels)

    def bits(self, decoder, idx, device):
        """-log2 p(x | code) for images idx [b]."""
        logits = decoder(self.deltas[idx].to(device).float(), self.sides[idx].to(device), self.ms[idx].to(device),
                         self.mu[idx].to(device))
        return F.cross_entropy(logits, self.levels[idx].to(device).long(), reduction="none").sum(1) / math.log(2)


@torch.no_grad()
def evaluate(decoder, leaves, device, batch):
    """-log2 p(x | code) for every image [N], on the CPU."""
    decoder.eval()
    idx = torch.arange(len(leaves))
    return torch.cat([leaves.bits(decoder, idx[i:i + batch], device).cpu() for i in range(0, len(leaves), batch)])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--codes", required=True, help="folder with dump_autoencoder.py's train.pt, val.pt and test.pt")
    p.add_argument("--tokens", type=int, default=None, help="code length (default: all the codes' tokens)")
    p.add_argument("--prior", default=None, help="train_prior.py's checkpoint, to price the codes (default: the "
                                                 "autoencoder's own shares)")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--channels", type=int, default=64, help="width of the mean image's convolutions")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--out", default=None, help="checkpoint (default: decoder.pt in the run's artifacts folder)")
    p.add_argument("--results", default=None, help="results prefix (default: results in the run's artifacts folder)")
    p.add_argument("--log", default=None, help="summary log (default: run.log in the run's artifacts folder)")
    p.add_argument("--plot", default=None, help="progress plot, redrawn every epoch (default: --log's path as .png)")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="data")
    args = p.parse_args()
    device = get_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    out, results = args.out or artifact("decoder.pt"), args.results or artifact("results")
    args.log = args.log or artifact("run.log")
    args.plot = args.plot or os.path.splitext(args.log)[0] + ".png"
    log = RunLog(args.log, device)
    log(f"=== train_categorical_decoder.py {vars(args)}")

    began = time.time()
    codes = {s: load_codes(os.path.join(args.codes, f"{s}.pt")) for s in ("train", "val", "test")}
    K = args.tokens or codes["train"].depths
    assert K <= codes["train"].depths, f"--tokens {K}: the codes have {codes['train'].depths}"
    leaves = {s: Leaves(c, load_mnist(s, args.data_root)[:len(c)], K) for s, c in codes.items()}
    prior = load_prior(torch.load(args.prior, map_location=device, weights_only=False), device) if args.prior else None
    code_bits = {}
    for s, c in codes.items():  # each image's code's bits, under the prior or the autoencoder's own shares
        if prior is not None:
            code_bits[s] = token_bits(prior, c.rare, c.m, c.r, device)[:, :K].sum(1)
        else:
            code_bits[s] = bits_of(own_shares(c.m[:, :K].float()), c.rare[:, :K]).sum(1)
        mse = (leaves[s].mu - leaves[s].levels.float().add(0.5).div(256)).square().mean().item()
        log(f"{s}: {len(c)} codes of {K} tokens, {code_bits[s].mean().item():.2f} bits each "
            f"({'prior' if prior is not None else 'own shares'}), mean image mse {mse:.5f}")
    log(f"loaded in {time.time() - began:.0f}s")

    torch.manual_seed(args.seed)
    decoder = LeafDecoder(K, args.width, args.layers, args.heads, args.channels).to(device)
    log(f"categorical decoder: {sum(q.numel() for q in decoder.parameters()) / 1e6:.1f}M parameters")
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gen = torch.Generator().manual_seed(args.seed)
    best: tuple[float, int, dict | None] = (math.inf, -1, None)
    history = []
    N = len(leaves["train"])
    for epoch in range(args.epochs):
        t0 = time.time()
        decoder.train()
        total = 0.0
        order = torch.randperm(N, generator=gen)
        for i in range(0, N, args.batch):
            bits = leaves["train"].bits(decoder, order[i:i + args.batch], device)
            loss = bits.mean() / D
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            opt.step()
            total += bits.sum().item()
        t1 = time.time()
        val = evaluate(decoder, leaves["val"], device, args.batch).mean().item() / D
        history.append(dict(epoch=epoch, val=val, train=total / (N * D), seconds=time.time() - t0))
        plot_categorical(history, args.plot)
        improved = val < best[0]
        log(f"  epoch {epoch:2d}: val {val:.4f} pixel bits/dim (with the code "
            f"{val + code_bits['val'].mean().item() / D:.4f}); train {total / (N * D):.4f} "
            f"({t1 - t0:.0f}s train, {time.time() - t1:.0f}s val){'  *' if improved else ''}")
        if improved:
            best = (val, epoch, {key: v.detach().clone() for key, v in decoder.state_dict().items()})
            torch.save(dict(config=decoder.config, state_dict=best[2], codes=args.codes,
                            autoencoder=codes["train"].ckpt, tokens=K, args=vars(args), history=history), out)
        elif epoch - best[1] >= args.patience:
            break
    assert best[2] is not None
    decoder.load_state_dict(best[2])
    log(f"kept epoch {best[1]} ({best[0]:.4f} val pixel bits/dim), saved {out}")

    rows = {}
    for s in ("val", "test"):
        pixel = evaluate(decoder, leaves[s], device, args.batch)
        total = code_bits[s] + pixel
        rows[s] = dict(images=len(total), tokens=K, bpd=total.mean().item() / D,
                       nats=total.mean().item() * math.log(2), code_bits=code_bits[s].mean().item(),
                       pixel_bpd=pixel.mean().item() / D)
        r = rows[s]
        log(f"{s}: {r['bpd']:.4f} bits/dim ({r['nats']:.1f} nats/image): code {r['code_bits']:.2f} bits, "
            f"pixels {r['pixel_bpd']:.4f} bits/dim, over {r['images']} images")
    os.makedirs(os.path.dirname(results) or ".", exist_ok=True)
    priced = args.prior or "the autoencoder's own shares"
    with open(results + ".json", "w") as f:
        json.dump(dict(codes=args.codes, decoder=out, code_priced_by=priced,
                       bound="one code: -log2 p(z(x)) - log2 p(x | z(x))", **rows), f, indent=2)
    with open(results + ".md", "w") as f:
        f.write(f"Codes priced by {priced}.\n\n")
        f.write("| split | tokens | bpd | nats/image | code bits | pixel bpd |\n|---|---|---|---|---|---|\n")
        for s, r in rows.items():
            f.write(f"| {s} | {r['tokens']} | {r['bpd']:.4f} | {r['nats']:.1f} | {r['code_bits']:.2f} | "
                    f"{r['pixel_bpd']:.4f} |\n")
    log(f"wrote {results}.json and .md")


if __name__ == "__main__":
    main()
