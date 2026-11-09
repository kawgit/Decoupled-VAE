"""Test bits per dimension of the whole model: each test image scored as -log2 p(code) - log2 p(x | code), its
code (the encoder's, from dump_encoder.py) priced by the prior and its pixels by the decoder.
This is the one-code upper bound on -log2 p(x): the image's own code only, not summed over every code that could
have drawn it. With --prior, the code is also priced by the encoder's implied shares (dvae/refine.py's) for
comparison; without, by those alone. The table goes to stdout and to benchmark.md and .json in the run's
artifacts folder.

Run from the repo: python benchmark.py --decoder artifacts/<job>/decoder.pt --prior artifacts/<prior job>/prior.pt
"""

import argparse
import json
import math
import os

import torch

from dvae.artifacts import artifact
from dvae.decoder import TokenLeaves, load_token_decoder, pixel_bits, token_code_bits
from dvae.codes import load_token_codes
from dvae.data import get_device, load_mnist
from dvae.deltatree import D
from dvae.prior import load_token_prior


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--decoder", required=True, help="train_decoder.py's checkpoint")
    p.add_argument("--prior", default=None, help="train_prior.py's checkpoint")
    p.add_argument("--codes", default=None, help="dump_encoder.py's folder (default: the decoder's)")
    p.add_argument("--split", default="test", choices=("test", "val"))
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--out", default=None, help="results prefix (default: benchmark in the run's artifacts folder)")
    p.add_argument("--device", default=None)
    p.add_argument("--data-root", default="data")
    args = p.parse_args()
    device = get_device(args.device)
    dckpt = torch.load(args.decoder, map_location=device, weights_only=False)
    decoder, K = load_token_decoder(dckpt, device), dckpt["tokens"]
    folder = args.codes or dckpt["codes"]
    codes = load_token_codes(os.path.join(folder, f"{args.split}.pt"))
    priors = {"the encoder's implied shares": None}
    if args.prior:
        priors = {f"the prior ({args.prior})": load_token_prior(torch.load(args.prior, map_location=device,
                                                                     weights_only=False), device), **priors}
    priced = {name: token_code_bits(codes, K, prior, device) for name, prior in priors.items()}
    pixels = pixel_bits(decoder, TokenLeaves(codes, load_mnist(args.split, args.data_root)[:len(codes)], K), device,
                        args.batch)

    rows = []
    for name, bits in priced.items():
        total = bits + pixels
        rows.append(dict(code_priced_by=name, bpd=total.mean().item() / D, nats=total.mean().item() * math.log(2),
                         code_bits=bits.mean().item(), pixel_bpd=pixels.mean().item() / D))
    head = (f"{args.split}: {len(codes)} images, codes of {K} tokens from {folder}, pixels by {args.decoder}; "
            "one-code upper bound on -log2 p(x)")
    table = ["| code priced by | bits/dim | nats/image | code bits | pixel bits/dim |", "|---|---|---|---|---|"]
    table += [f"| {r['code_priced_by']} | {r['bpd']:.4f} | {r['nats']:.1f} | {r['code_bits']:.2f} | "
              f"{r['pixel_bpd']:.4f} |" for r in rows]
    print(head)
    print("\n".join(table))
    out = args.out or artifact("benchmark")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + ".md", "w") as f:
        f.write(head + "\n\n" + "\n".join(table) + "\n")
    with open(out + ".json", "w") as f:
        json.dump(dict(split=args.split, images=len(codes), tokens=K, codes=folder, decoder=args.decoder,
                       bound="one code: -log2 p(z(x)) - log2 p(x | z(x))", rows=rows), f, indent=2)
    print(f"wrote {out}.md and .json")


if __name__ == "__main__":
    main()
