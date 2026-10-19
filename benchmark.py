"""Benchmark test-set likelihood of prior + decoder(s).

The code space is small enough to enumerate, so we report the exact negative log-likelihood
-log sum_z p(z) p(x|z). With --encoder we also report the negative ELBO under q(z|x) (any q gives a
valid bound, so --encoder-mode greedy/topp are fine too), split into distortion and rate.

Each decoder checkpoint is scored on the data variant it was trained for: "gray" on 8-bit MNIST,
"binary" on the test set binarized with a fixed seed. Numbers are given in nats/image and bits/dim.
"""

import argparse
import json
import math
import os

import torch

from dvae.common import add_common_args, add_mode_args, check_same_codes, get_device, load_checkpoint
from dvae.data import NUM_PIXELS, batches, binary_target, gray_target, load_mnist, to_float

MAX_CODES = 1 << 16


def all_codes(V, k_max, device):
    """[V**k_max, k_max] in the tree's node order (z_1 most significant)."""
    idx = torch.arange(V**k_max, device=device)
    powers = V ** torch.arange(k_max - 1, -1, -1, device=device)
    return (idx[:, None] // powers) % V


@torch.no_grad()
def decoder_log_likelihoods(decoder, codes, target, code_chunk, image_chunk):
    """log p(x|z) for every test image and code -> [N, C]."""
    out = torch.empty(len(target), len(codes), device=target.device)
    pix = torch.arange(NUM_PIXELS, device=target.device)[None]
    for c0 in range(0, len(codes), code_chunk):
        logits = decoder(codes[c0:c0 + code_chunk])
        if decoder.likelihood == "gray":
            logp = logits.log_softmax(-1)  # [c, 784, 256]
            for i0 in range(0, len(target), image_chunk):
                t = target[i0:i0 + image_chunk]
                out[i0:i0 + image_chunk, c0:c0 + code_chunk] = logp[:, pix, t].sum(-1).T
        else:
            lp1, lp0 = torch.nn.functional.logsigmoid(logits), torch.nn.functional.logsigmoid(-logits)
            out[:, c0:c0 + code_chunk] = target @ lp1.T + (1 - target) @ lp0.T
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--prior", default="checkpoints/prior.pt")
    p.add_argument("--decoder", nargs="+", default=["checkpoints/decoder_gray.pt"])
    p.add_argument("--encoder", default=None, help="optional: also report the ELBO under this q(z|x)")
    add_mode_args(p, "encoder")
    p.add_argument("--binarize-seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="score only the first N test images")
    p.add_argument("--code-chunk", type=int, default=64)
    p.add_argument("--image-chunk", type=int, default=500)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    device = get_device(args.device)
    prior, _ = load_checkpoint(args.prior, "prior", device)
    V, k_max = prior.config["vocab_size"], prior.config["k_max"]
    if V**k_max > MAX_CODES:
        raise SystemExit(f"{V}**{k_max} codes is too many to enumerate exactly")
    codes = all_codes(V, k_max, device)
    test = load_mnist("test", args.data_root, device)
    if args.limit:
        test = test[: args.limit]
    N = len(test)

    with torch.no_grad():
        log_pz = prior.prefix_logprobs()[k_max]  # [C]
        log_q = None
        if args.encoder:
            encoder, _ = load_checkpoint(args.encoder, "encoder", device)
            check_same_codes(encoder.config, prior.config, "encoder vs prior")
            log_q = torch.cat([encoder.prefix_logprobs(to_float(test[idx]), args.encoder_mode, args.encoder_top_p)[k_max]
                               for idx in batches(N, 128, False, device, drop_last=False)])

    rows = []
    for path in args.decoder:
        decoder, _ = load_checkpoint(path, "decoder", device)
        check_same_codes(decoder.config, prior.config, f"{path} vs prior")
        if decoder.likelihood == "gray":
            variant, target = "mnist-8bit", gray_target(test)
        else:
            gen = torch.Generator(device=device).manual_seed(args.binarize_seed)
            variant, target = f"mnist-binarized(seed={args.binarize_seed})", binary_target(test, gen)
        log_px_z = decoder_log_likelihoods(decoder, codes, target, args.code_chunk, args.image_chunk)  # [N, C]
        log_joint = log_px_z + log_pz
        nll = -torch.logsumexp(log_joint, 1)  # exact -log p(x), [N]
        row = {"decoder": path, "variant": variant, "num_images": N, "nll_nats": nll.mean().item()}
        if log_q is not None:
            q = log_q.exp()
            safe_log_q = torch.where(q > 0, log_q, torch.zeros_like(log_q))
            distortion = -(q * torch.where(q > 0, log_px_z, torch.zeros_like(log_px_z))).sum(1)
            rate = (q * (safe_log_q - torch.where(q > 0, log_pz, torch.zeros_like(log_q)))).sum(1)  # KL(q || p)
            row.update(neg_elbo_nats=(distortion + rate).mean().item(), distortion_nats=distortion.mean().item(),
                       rate_nats=rate.mean().item(), encoder=args.encoder, encoder_mode=args.encoder_mode)
        for key in [k for k in row if k.endswith("_nats")]:
            row[key.replace("_nats", "_bpd")] = row[key] / (NUM_PIXELS * math.log(2))
        rows.append(row)
        print(json.dumps(row, indent=2))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "benchmark.json"), "w") as f:
        json.dump({"prior": args.prior, "codes": f"{V}^{k_max}", "results": rows}, f, indent=2)
    lines = ["| variant | NLL (nats/img) | NLL (bpd) | -ELBO (nats/img) | -ELBO (bpd) | rate (nats) | baseline |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        elbo = (f"{r['neg_elbo_nats']:.2f} | {r['neg_elbo_bpd']:.4f} | {r['rate_nats']:.2f}"
                if "neg_elbo_nats" in r else "– | – | –")
        lines.append(f"| {r['variant']} | {r['nll_nats']:.2f} | {r['nll_bpd']:.4f} | {elbo} | |")
    with open(os.path.join(args.out_dir, "benchmark.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out_dir}/benchmark.json and benchmark.md")


if __name__ == "__main__":
    main()
