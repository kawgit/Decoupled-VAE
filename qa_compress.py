"""Human QA of compression: encode test images, keep the first k tokens, let the prior complete the
rest, decode. For each k, saves a grid where each row is [original | m decompressed completions].
"""

import argparse
import os

import torch

from dvae.common import add_common_args, add_mode_args, check_encoder, check_same_codes, get_device, load_checkpoint
from dvae.data import load_mnist, to_float
from dvae.viz import code_str, save_grid


def default_ks(k_max):
    if k_max <= 10:
        return list(range(1, k_max + 1))
    return sorted({2**i for i in range(k_max.bit_length()) if 2**i < k_max} | {k_max})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--encoder", default="checkpoints/encoder.pt")
    p.add_argument("--prior", default="checkpoints/prior.pt")
    p.add_argument("--decoder", default="checkpoints/decoder_gray.pt")
    p.add_argument("-n", type=int, default=16, help="number of test images")
    p.add_argument("-k", type=int, nargs="+", default=None,
                   help="prefix lengths to keep (default: every k up to the prior's k_max if it's at most 10, "
                        "else powers of 2 and k_max)")
    p.add_argument("-m", type=int, default=8, help="prior completions per image")
    add_mode_args(p, "encoder", default="greedy")
    add_mode_args(p, "prior")
    p.add_argument("--render", choices=["mean", "sample", "mode"], default="mean")
    p.add_argument("--out-dir", default="outputs/compress")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    encoder, _ = load_checkpoint(args.encoder, "encoder", device)
    prior, _ = load_checkpoint(args.prior, "prior", device)
    decoder, _ = load_checkpoint(args.decoder, "decoder", device)
    check_same_codes(prior.config, decoder.config, "prior vs decoder")
    check_encoder(encoder, prior.config, "prior")
    k_max = prior.config["k_max"]
    ks = args.k or default_ks(k_max)
    if any(k < 0 or k > k_max for k in ks):
        print(f"skipping k outside 0..{k_max}: {[k for k in ks if k < 0 or k > k_max]}")
        ks = [k for k in ks if 0 <= k <= k_max]
        if not ks:
            raise SystemExit("no valid -k left")

    test = load_mnist("test", args.data_root, device)
    pick = torch.randperm(len(test), generator=torch.Generator().manual_seed(args.seed))[: args.n].to(device)
    x = to_float(test[pick])
    with torch.no_grad():
        codes = encoder.sample(x, k_max, mode=args.encoder_mode, top_p=args.encoder_top_p)
        for k in ks:
            prefix = codes[:, :k].repeat_interleave(args.m, 0)
            full = prior.sample(len(prefix), encoder, prefix=prefix, mode=args.prior_mode, top_p=args.prior_top_p)
            recon = decoder.render(decoder(full, encoder.slice_path(full)), args.render).view(args.n, args.m, 1, 28, 28)
            grid = torch.cat([x.unsqueeze(1), recon], 1).flatten(0, 1)
            path = os.path.join(args.out_dir, f"k{k:02d}.png")
            save_grid(grid, path, nrow=args.m + 1)
            print(f"k={k}: saved {path}")
    print("encoder codes (one per row):")
    for i, c in enumerate(codes):
        print(f"  {i:3d} {code_str(c)}")


if __name__ == "__main__":
    main()
