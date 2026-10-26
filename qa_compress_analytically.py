"""Human QA of compression with analytical inversion: encode test images, keep the first k tokens, and
decompress by drawing training images from the ideal decoder's posterior p(y | z_1:k) ∝ q(z_1:k | y), the
posterior in the decompression loss. No prior or decoder is involved. For each k, saves a grid where each
row is [original | m decompressed images].
"""

import argparse
import os

import torch

from dvae.common import add_common_args, add_mode_args, get_device, load_checkpoint
from dvae.data import load_mnist, to_float
from dvae.viz import code_str, save_grid
from qa_compress import default_ks

CHUNK = 8192  # training images per forward pass


@torch.no_grad()
def prefix_log_q(encoder, Y, codes, mode, top_p):
    """log q(z_1:t | y) of every training image y [N, 1, 28, 28] for each code's prefixes: entry t is [n, N].

    Every slice's output on the training images is computed once and shared by all the codes.
    """
    logp = torch.cat([encoder.answers(Y[i:i + CHUNK], mode, top_p) for i in range(0, len(Y), CHUNK)])  # [N, S+1, V]
    path = encoder.slice_path(codes)  # the slice asked at every position of each code
    out = [torch.zeros(len(codes), len(Y), device=Y.device)]
    for t in range(codes.shape[1]):
        out.append(out[-1] + logp[:, path[:, t], codes[:, t]].T)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--encoder", default="checkpoints/encoder.pt")
    p.add_argument("-n", type=int, default=16, help="number of test images")
    p.add_argument("--k-max", type=int, default=32, help="code length")
    p.add_argument("-k", type=int, nargs="+", default=None,
                   help="prefix lengths to keep (default: every k up to k_max if it's at most 10, else powers of 2 "
                        "and k_max)")
    p.add_argument("-m", type=int, default=8, help="posterior draws per image")
    add_mode_args(p, "encoder", default="greedy")
    add_mode_args(p, "posterior")
    p.add_argument("--out-dir", default="outputs/compress_analytically")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    encoder, _ = load_checkpoint(args.encoder, "encoder", device)
    k_max = args.k_max
    ks = args.k or default_ks(k_max)
    if any(k < 0 or k > k_max for k in ks):
        print(f"skipping k outside 0..{k_max}: {[k for k in ks if k < 0 or k > k_max]}")
        ks = [k for k in ks if 0 <= k <= k_max]
        if not ks:
            raise SystemExit("no valid -k left")

    test = load_mnist("test", args.data_root, device)
    train = to_float(load_mnist("train", args.data_root, device))
    pick = torch.randperm(len(test), generator=torch.Generator().manual_seed(args.seed))[: args.n].to(device)
    x = to_float(test[pick])
    gen = torch.Generator().manual_seed(args.seed + 1)
    with torch.no_grad():
        codes = encoder.sample(x, k_max, mode=args.encoder_mode, top_p=args.encoder_top_p)
        log_q = prefix_log_q(encoder, train, codes[:, :max(ks)], args.posterior_mode, args.posterior_top_p)
        for k in ks:
            lq = log_q[k]
            lq = torch.where(lq.amax(1, keepdim=True) > -torch.inf, lq, 0.0)  # no training image fits: uniform
            post = lq.softmax(1).cpu()  # [n, N]: p(y | z_1:k) over the training images
            draws = torch.multinomial(post, args.m, replacement=True, generator=gen).to(device)
            grid = torch.cat([x.unsqueeze(1), train[draws]], 1).flatten(0, 1)
            path = os.path.join(args.out_dir, f"k{k:02d}.png")
            save_grid(grid, path, nrow=args.m + 1)
            effective = (1 / post.pow(2).sum(1)).median().item()
            print(f"k={k}: saved {path} (posterior spreads over a median of {effective:.1f} training images)")
    print("encoder codes (one per row):")
    for i, c in enumerate(codes):
        print(f"  {i:3d} {code_str(c)}")


if __name__ == "__main__":
    main()
