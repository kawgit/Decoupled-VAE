"""Stage 3: train the decoder (dvae/refine.py's TokenDecoder), p(x | code): 256 grey levels per
pixel given a code's first --tokens tokens, by maximum likelihood on the training images, on the codes a
train_encoder_components.py run keeps in its tree/ folder.

It reads a code's delta history and its node's predicted mean image (the training mean plus the deltas), so it
adds what the encoder's mean leaves out: each pixel's spread around it. Every epoch logs and plots
(dvae/plots.py's plot_decoder) the val and train images' pixel bits per dimension under it, and early
stopping watches val (dvae/stopping.py's patience). Then val and test are scored as -log2 p(code) - log2 p(x | code), the one-code upper bound
on -log2 p(x): the code priced by --prior (train_prior.py's) when given, else by the encoder's implied shares.

Run from the repo: python train_decoder.py --tree artifacts/<encoder job>
                   [--prior artifacts/<prior job>/prior.pt]
"""

import argparse
import json
import math
import os
import time

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")  # ~6.5 GB: fail instead of swapping
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")

import torch  # noqa: E402

from dvae.artifacts import artifact  # noqa: E402
from dvae.decoder import TokenLeaves, pixel_bits, token_code_bits as price  # noqa: E402
from dvae.codes import TreeCodes  # noqa: E402
from dvae.data import SHAPES, get_device, load_images  # noqa: E402
from dvae.logs import RunLog  # noqa: E402
from dvae.plots import plot_decoder  # noqa: E402
from dvae.prior import load_token_prior  # noqa: E402
from dvae.refine import TokenDecoder  # noqa: E402
from dvae.stopping import add_patience_args, stagnant  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tree", required=True,
                   help="a train_encoder_components.py run's folder, whose tree/ holds the codes")
    p.add_argument("--tokens", type=int, default=None, help="code length (default: all the codes' tokens)")
    p.add_argument("--prior", default=None, help="train_prior.py's checkpoint, to price the codes (default: the "
                                                 "encoder's implied shares)")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--channels", type=int, default=64, help="width of the mean image's convolutions")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=50)
    add_patience_args(p)
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
    log(f"=== train_decoder.py {vars(args)}")

    began = time.time()
    codes = {s: TreeCodes(args.tree, s) for s in ("train", "val", "test")}
    D = int(codes["train"].mean.numel())
    dataset = next((name for name, sh in SHAPES.items() if sh[0] * sh[1] * sh[2] == D), "mnist")
    K = args.tokens or codes["train"].depths
    assert K <= codes["train"].depths, f"--tokens {K}: the codes have {codes['train'].depths}"
    prior = load_token_prior(torch.load(args.prior, map_location=device, weights_only=False), device) if args.prior else None
    code_bits, leaves = {}, {}
    for s, c in codes.items():  # each image's code's bits, under the prior or the encoder's implied shares
        code_bits[s] = price(c, K, prior, device)
        leaves[s] = TokenLeaves(c, load_images(dataset, s, args.data_root)[:len(c)], K)
        mse = (leaves[s].mu - leaves[s].levels.float().add(0.5).div(256)).square().mean().item()
        log(f"{s}: {len(c)} codes of {K} tokens, {code_bits[s].mean().item():.2f} bits each "
            f"({'prior' if prior is not None else 'implied shares'}), mean image mse {mse:.5f}")
    log(f"loaded in {time.time() - began:.0f}s")

    torch.manual_seed(args.seed)
    decoder = TokenDecoder(K, codes["train"].V, args.width, args.layers, args.heads, args.channels,
                           SHAPES[dataset]).to(device)
    log(f"decoder: {sum(q.numel() for q in decoder.parameters()) / 1e6:.1f}M parameters")
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
        val = pixel_bits(decoder, leaves["val"], device, args.batch).mean().item() / D
        history.append(dict(epoch=epoch, val=val, train=total / (N * D), seconds=time.time() - t0))
        plot_decoder(history, args.plot)
        improved = val < best[0]
        log(f"  epoch {epoch:2d}: val {val:.4f} pixel bits/dim (with the code "
            f"{val + code_bits['val'].mean().item() / D:.4f}); train {total / (N * D):.4f} "
            f"({t1 - t0:.0f}s train, {time.time() - t1:.0f}s val){'  *' if improved else ''}")
        if improved:
            best = (val, epoch, {key: v.detach().clone() for key, v in decoder.state_dict().items()})
            torch.save(dict(config=decoder.config, state_dict=best[2], codes=args.tree,
                            encoder=codes["train"].ckpt, tokens=K, args=vars(args), history=history), out)
        elif stagnant(epoch, best[1], args):
            break
    assert best[2] is not None
    decoder.load_state_dict(best[2])
    log(f"kept epoch {best[1]} ({best[0]:.4f} val pixel bits/dim), saved {out}")

    rows = {}
    for s in ("val", "test"):
        pixel = pixel_bits(decoder, leaves[s], device, args.batch)
        total = code_bits[s] + pixel
        rows[s] = dict(images=len(total), tokens=K, bpd=total.mean().item() / D,
                       nats=total.mean().item() * math.log(2), code_bits=code_bits[s].mean().item(),
                       pixel_bpd=pixel.mean().item() / D)
        r = rows[s]
        log(f"{s}: {r['bpd']:.4f} bits/dim ({r['nats']:.1f} nats/image): code {r['code_bits']:.2f} bits, "
            f"pixels {r['pixel_bpd']:.4f} bits/dim, over {r['images']} images")
    os.makedirs(os.path.dirname(results) or ".", exist_ok=True)
    priced = args.prior or "the encoder's implied shares"
    with open(results + ".json", "w") as f:
        json.dump(dict(codes=args.tree, decoder=out, code_priced_by=priced,
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
