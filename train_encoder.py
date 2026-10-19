"""Stage 1: train the encoder q(z|x) alone, with the ideal-decoder objective (see dvae/objective.py).

The code space is small enough (V**k_max) to enumerate, so the objective and its gradient are exact
given the reference pool; no REINFORCE needed.
"""

import argparse

import torch

from dvae.common import add_common_args, add_train_args, fit, get_device, save_checkpoint
from dvae.data import batches, load_mnist, to_float
from dvae.models import Encoder
from dvae.objective import encoder_objective, k_weights


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    add_train_args(p, out="checkpoints/encoder.pt", batch_size=None, lr=3e-4, epochs=30, patience=3)
    p.add_argument("--num-refs", type=int, default=256, help="D: images per step used to estimate the posterior")
    p.add_argument("--vocab-size", type=int, default=2)
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--k-dist", choices=["uniform", "geometric", "custom"], default="uniform")
    p.add_argument("--k-geom-ratio", type=float, default=0.7)
    p.add_argument("--k-weights", default=None, help="comma-separated p(k) for --k-dist custom")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-img-layers", type=int, default=2)
    p.add_argument("--n-tok-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--max-steps-per-epoch", type=int, default=None, help="for smoke tests")
    args = p.parse_args()
    args.batch_size = args.batch_size or args.num_refs  # B anchors, taken from the D-image pool
    assert args.batch_size <= args.num_refs, "anchors are a subset of the pool, so --batch-size <= --num-refs"

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    print(f"device {device}")
    train, val = load_mnist("train", args.data_root, device), load_mnist("val", args.data_root, device)
    model = Encoder(args.vocab_size, args.k_max, args.d_model, args.n_img_layers, args.n_tok_layers, args.n_heads).to(device)
    print(f"encoder params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M, codes {args.vocab_size ** args.k_max}")
    weights = k_weights(args.k_max, args.k_dist, args.k_geom_ratio, args.k_weights).to(device)
    gen = torch.Generator().manual_seed(args.seed)

    def step(images, n_anchors):
        x = to_float(images)
        return encoder_objective(model.prefix_logprobs(x), x.flatten(1), n_anchors, weights)

    def train_epoch(epoch):
        for i, idx in enumerate(batches(len(train), args.num_refs, True, device, gen)):
            if args.max_steps_per_epoch and i >= args.max_steps_per_epoch:
                break
            yield step(train[idx], args.batch_size)[0]

    def evaluate():
        # Fixed pools of D val images, every image an anchor, so numbers are comparable across epochs/runs.
        totals = None
        n = 0
        for idx in batches(len(val), args.num_refs, False, device):
            loss, per_k, k0 = step(val[idx], args.num_refs)
            vals = torch.cat([loss[None], k0[None], per_k])
            totals = vals if totals is None else totals + vals
            n += 1
        totals = (totals / n).tolist()
        return {"loss": totals[0], "k0": totals[1], "per_k": totals[2:]}

    def save(metrics, history):
        save_checkpoint(args.out, "encoder", model, args=vars(args), metrics=metrics, history=history)

    fit(model, args, train_epoch, evaluate, save)
    print(f"saved best encoder to {args.out}")


if __name__ == "__main__":
    main()
