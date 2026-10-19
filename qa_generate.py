"""Human QA of generation: sample codes from the prior, decode them, save a grid of n images."""

import argparse
import math

import torch

from dvae.common import add_common_args, add_mode_args, check_same_codes, get_device, load_checkpoint
from dvae.viz import code_str, save_grid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--prior", default="checkpoints/prior.pt")
    p.add_argument("--decoder", default="checkpoints/decoder_gray.pt")
    p.add_argument("-n", type=int, default=64)
    add_mode_args(p, "prior")
    p.add_argument("--render", choices=["mean", "sample", "mode"], default="mean",
                   help="show the decoder's per-pixel mean, a pixel sample, or the per-pixel mode")
    p.add_argument("--out", default="outputs/generate.png")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    prior, _ = load_checkpoint(args.prior, "prior", device)
    decoder, _ = load_checkpoint(args.decoder, "decoder", device)
    check_same_codes(prior.config, decoder.config, "prior vs decoder")

    with torch.no_grad():
        codes = prior.sample(args.n, mode=args.prior_mode, top_p=args.prior_top_p)
        images = decoder.render(decoder(codes), args.render)
    save_grid(images, args.out, nrow=math.ceil(math.sqrt(args.n)))
    print(f"saved {args.n} samples to {args.out}")
    print("codes (row-major):")
    for i, c in enumerate(codes):
        print(f"  {i:3d} {code_str(c)}")


if __name__ == "__main__":
    main()
