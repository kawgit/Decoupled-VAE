"""How much of the latent space does an encoder use?

Computed exactly over a data split by enumerating the code tree, for every prefix length k up to k_max or the
deepest whose V**k codes can be enumerated:

  capacity     k * log2(V) bits: what the first k tokens could hold
  H[q̄]         entropy of the aggregate posterior q̄(z_1:k) = E_x q(z_1:k|x): bits the code space spreads over
  H[q|x]       average entropy of q(z_1:k|x): encoder noise, bits that don't depend on the image
  I(x;z)       H[q̄] - H[q|x]: bits about the image, i.e. how much of the capacity is actually used
  +I           what token k adds over the first k-1
  codes        distinct most-likely prefixes across the split, out of V**k
  eff          2**H[q̄], the effective number of codes
  I(y;z)       bits about the digit label (out of H[y] ~ 3.32), to see whether early tokens carry class
"""

import argparse
import json
import math
import os

import torch

from dvae.common import add_common_args, add_mode_args, get_device, load_checkpoint
from dvae.data import batches, load_labels, load_mnist, to_float

MAX_CODES = 1 << 16


def entropy_bits(p, dim=-1):
    return -(p * torch.where(p > 0, p.log2(), torch.zeros_like(p))).sum(dim)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--encoder", default="checkpoints/encoder.pt")
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    add_mode_args(p, "encoder")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", default=None, help="JSON output (default: results/usage_<encoder name>.json)")
    args = p.parse_args()

    device = get_device(args.device)
    encoder, _ = load_checkpoint(args.encoder, "encoder", device)
    V, k_max = encoder.config["vocab_size"], encoder.config["k_max"]
    depth = max((t for t in range(1, k_max + 1) if V**t <= MAX_CODES), default=0)
    if depth == 0:
        raise SystemExit(f"{V} codes per token is too many to enumerate exactly")
    if depth < k_max:
        print(f"note: {V}**{depth + 1} codes is too many to enumerate exactly, so only prefixes up to "
              f"k={depth} of {k_max} are analyzed")
    images = load_mnist(args.split, args.data_root, device)
    labels = load_labels(args.split, args.data_root, device)
    N, n_labels = len(images), 10

    widths = [V**t for t in range(depth + 1)]
    agg = [torch.zeros(w, device=device) for w in widths]
    agg_by_label = [torch.zeros(n_labels, w, device=device) for w in widths]
    cond_entropy = torch.zeros(depth + 1, device=device)
    used = [torch.zeros(w, dtype=torch.bool, device=device) for w in widths]
    with torch.no_grad():
        for idx in batches(N, args.batch_size, False, device, drop_last=False):
            tree = encoder.prefix_logprobs(to_float(images[idx]), args.encoder_mode, args.encoder_top_p,
                                           depth=depth)
            for t in range(1, depth + 1):
                q = tree[t].exp()  # [n, V**t]
                agg[t] += q.sum(0)
                agg_by_label[t].index_add_(0, labels[idx], q)
                cond_entropy[t] += entropy_bits(q).sum()
                used[t][q.argmax(1)] = True

    label_counts = torch.bincount(labels, minlength=n_labels).float()
    p_label = label_counts / N
    rows, prev_info = [], 0.0
    for t in range(1, depth + 1):
        h_agg = entropy_bits(agg[t] / N).item()
        h_cond = (cond_entropy[t] / N).item()
        info = h_agg - h_cond
        h_given_label = (p_label * entropy_bits(agg_by_label[t] / label_counts[:, None])).sum().item()
        capacity = t * math.log2(V)
        rows.append(dict(k=t, capacity_bits=capacity, agg_entropy_bits=h_agg, cond_entropy_bits=h_cond,
                         info_bits=info, utilization=info / capacity, info_added_bits=info - prev_info,
                         codes_used=int(used[t].sum().item()), codes_total=widths[t], effective_codes=2**h_agg,
                         label_info_bits=h_agg - h_given_label))
        prev_info = info

    label_entropy = entropy_bits(p_label).item()
    print(f"{args.encoder}: V={V}, k_max={k_max} (analyzed to k={depth}), {args.split} split ({N} images), "
          f"encoder mode {args.encoder_mode}")
    header = f"{'k':>3} {'capacity':>8} {'H[q̄]':>7} {'H[q|x]':>7} {'I(x;z)':>7} {'used':>6} {'+I':>6} {'codes':>13} {'eff':>8} {'I(y;z)':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['k']:>3} {r['capacity_bits']:>8.2f} {r['agg_entropy_bits']:>7.2f} {r['cond_entropy_bits']:>7.2f} "
              f"{r['info_bits']:>7.2f} {100 * r['utilization']:>5.1f}% {r['info_added_bits']:>6.2f} "
              f"{r['codes_used']:>6}/{r['codes_total']:<6} {r['effective_codes']:>8.1f} {r['label_info_bits']:>7.2f}")
    last = rows[-1]
    print(f"\nuses {last['info_bits']:.2f} of {last['capacity_bits']:.2f} bits ({100 * last['utilization']:.1f}%); "
          f"label: {last['label_info_bits']:.2f} of {label_entropy:.2f} bits")

    out = args.out or os.path.join("results", f"usage_{os.path.splitext(os.path.basename(args.encoder))[0]}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"encoder": args.encoder, "vocab_size": V, "k_max": k_max, "analyzed_depth": depth,
                   "split": args.split, "num_images": N,
                   "encoder_mode": args.encoder_mode, "label_entropy_bits": label_entropy, "per_k": rows}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
