"""Stage 1: train the encoder q(z|x) alone, with the ideal-decoder objective (see dvae/objective.py).

Two estimators of the same objective:

exact    Every step draws a pool of D = --num-refs images; the first B = --batch-size are anchors and
         each anchor's posterior is over the rest of the pool. Every prefix of every code is scored,
         so the objective and its gradient are exact given the pool. Cost grows with the code tree.
sampled  Every step takes B anchors and D separate random reference images. S = --samples codes are
         drawn from each anchor's q and scored under every reference; the gradient is backprop through
         the posterior plus REINFORCE for the sampling. Cost grows with B * S * D * k_max, not with V.

--estimator auto picks exact when its estimated memory fits under --max-mem-gb, else sampled.
"""

import argparse

import torch

from dvae.common import add_common_args, add_train_args, fit, get_device, parse_train_args
from dvae.data import batches, load_mnist, to_float
from dvae.models import Encoder
from dvae.objective import encoder_objective, k_weights, sampled_encoder_objective

DEFAULT_SAMPLED_BATCH = 32


def per_position_bytes(args):
    # Calibrated on MPS: ~5.5 GB for 256 images x 1023 tree nodes at d_model=64 with 2 token layers.
    return 21e3 * (args.d_model / 64) * (args.n_tok_layers / 2)


def exact_memory_gb(args):
    V, k_max, D = args.vocab_size, args.k_max, args.num_refs
    nodes = k_max if V == 1 else (V**k_max - 1) // (V - 1)
    return (D * nodes * per_position_bytes(args) + 10 * 4 * D * V**k_max) / 1e9, nodes


def sampled_memory_gb(args):
    pairs = args.batch_size * args.samples * args.num_refs
    return pairs * args.k_max * (per_position_bytes(args) + 3 * 4 * args.vocab_size) / 1e9


def choose_estimator(args):
    V, k_max = args.vocab_size, args.k_max
    exact_gb, nodes = exact_memory_gb(args)
    if args.estimator == "auto":
        args.estimator = "exact" if exact_gb <= args.max_mem_gb else "sampled"
    if args.estimator == "exact":
        args.batch_size = args.batch_size or args.num_refs
        assert args.batch_size <= args.num_refs, "exact: anchors are a subset of the pool, so --batch-size <= --num-refs"
        print(f"estimator exact: {V**k_max} codes, {nodes} prefix nodes per image, ~{exact_gb:.1f} GB per step")
        if exact_gb > args.max_mem_gb:
            raise SystemExit(f"V={V}, k_max={k_max} needs ~{exact_gb:.0f} GB to enumerate with --num-refs "
                             f"{args.num_refs} (> --max-mem-gb {args.max_mem_gb:g}); use --estimator sampled")
    else:
        args.batch_size = args.batch_size or DEFAULT_SAMPLED_BATCH
        gb = sampled_memory_gb(args)
        print(f"estimator sampled: {V**k_max} codes, B={args.batch_size} anchors x S={args.samples} samples x "
              f"D={args.num_refs} refs, ~{gb:.1f} GB per step")
        if gb > args.max_mem_gb:
            raise SystemExit(f"~{gb:.0f} GB per step (> --max-mem-gb {args.max_mem_gb:g}); "
                             f"lower --batch-size, --samples or --num-refs")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    add_train_args(p, out="checkpoints/encoder.pt", batch_size=None, lr=3e-4, epochs=30, patience=3)
    p.add_argument("--estimator", choices=["auto", "exact", "sampled"], default="auto")
    p.add_argument("--num-refs", type=int, default=256, help="D: images per step used to estimate the posterior")
    p.add_argument("--samples", type=int, default=4, help="S: codes drawn per anchor (sampled estimator)")
    p.add_argument("--vocab-size", type=int, default=2)
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--k-dist", choices=["uniform", "geometric", "custom"], default="uniform")
    p.add_argument("--k-geom-ratio", type=float, default=0.7)
    p.add_argument("--k-weights", default=None, help="comma-separated p(k) for --k-dist custom")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-img-layers", type=int, default=2)
    p.add_argument("--n-tok-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--max-mem-gb", type=float, default=16.0,
                   help="refuse to start if the estimated activation memory per step exceeds this")
    args = parse_train_args(p)
    choose_estimator(args)

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    print(f"device {device}")
    train, val = load_mnist("train", args.data_root, device), load_mnist("val", args.data_root, device)
    model = Encoder(args.vocab_size, args.k_max, args.d_model, args.n_img_layers, args.n_tok_layers, args.n_heads).to(device)
    print(f"encoder params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    weights = k_weights(args.k_max, args.k_dist, args.k_geom_ratio, args.k_weights).to(device)
    gen = torch.Generator().manual_seed(args.seed)
    B, S, D = args.batch_size, args.samples, args.num_refs

    def exact_step(images, n_anchors):
        x = to_float(images)
        return encoder_objective(model.prefix_logprobs(x), x.flatten(1), n_anchors, weights)

    def sampled_step(data, anchor_idx, ref_idx, generator=None):
        xa, xr = to_float(data[anchor_idx]), to_float(data[ref_idx])
        memory = model.memory(xa).repeat_interleave(S, 0)
        codes = model.tokens.sample(len(memory), memory, generator=generator)  # [B*S, k_max]
        logq_self = model.tokens.log_prob(codes, memory).view(len(xa), S, -1)
        logq_refs = model.tokens.log_prob_pairwise(codes, model.memory(xr)).view(len(xa), S, len(xr), -1)
        self_mask = anchor_idx[:, None] == ref_idx[None, :]
        return sampled_encoder_objective(logq_self, logq_refs, xa.flatten(1), xr.flatten(1), self_mask, weights)

    def epoch_batches(epoch):
        if args.estimator == "exact":
            yield from batches(len(train), D, True, device, gen)
            return
        for anchor_idx in batches(len(train), B, True, device, gen):
            # drawn inside the generator so skipping steps on resume replays the same draws
            yield anchor_idx, torch.randperm(len(train), generator=gen)[:D].to(device)

    def step_loss(batch):
        if args.estimator == "exact":
            return exact_step(train[batch], B)[0]
        return sampled_step(train, *batch)[0]

    def evaluate():
        # Fixed anchors, references and (for sampled) code draws, so numbers are comparable across epochs.
        rows = []
        if args.estimator == "exact":
            for idx in batches(len(val), D, False, device):  # every pool image is an anchor
                loss, per_k, k0 = exact_step(val[idx], D)
                rows.append(torch.cat([loss[None], k0[None], per_k]))
        else:
            val_gen = torch.Generator(device=device).manual_seed(1234)
            ref_idx = torch.arange(D, device=device)
            for anchor_idx in batches(len(val), B, False, device):
                loss, per_k, k0 = sampled_step(val, anchor_idx, ref_idx, val_gen)
                rows.append(torch.cat([loss.detach()[None], k0[None], per_k]))
        totals = torch.stack(rows).mean(0).tolist()
        return {"loss": totals[0], "k0": totals[1], "per_k": totals[2:]}

    fit(model, "encoder", args, epoch_batches, step_loss, evaluate, gen)


if __name__ == "__main__":
    main()
