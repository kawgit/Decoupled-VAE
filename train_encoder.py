"""Stage 1: train the encoder q(z|x) alone, with the ideal-decoder objective (see dvae/objective.py).

Every step takes B anchors and a pool of D reference images (the anchors plus D - B extras).

--estimator      which gradient terms to keep: full (default), reinforce (own code choice only; the
                 reference scoring then runs without gradient, so it is much cheaper), pathwise
                 (backprop through the posterior only).
--num-samples S  codes per anchor, drawn from q' (q at --temperature) and importance-weighted back to q.
                 If S >= V**k_max, every code is enumerated instead and the objective is exact.
--ref-ranking    the ranking turned into posterior', the distribution extra references are drawn from:
                 uniform (default) or two-towers (a learned score image . prefix vector, one per prefix
                 length, mixed over k by p(k)). Corrected with importance weights; --ref-uniform-frac of
                 the extras stay uniform.

Validation always uses fixed pools of uniform val references and codes drawn from q itself, so runs
with different settings report comparable numbers.
"""

import argparse

import torch

from dvae.common import add_common_args, add_train_args, fit, get_device, parse_train_args
from dvae.data import batches, load_mnist, to_float
from dvae.models import Encoder
from dvae.objective import (ESTIMATORS, anchor_distances, combine, k_weights, sampled_posterior, sampled_terms,
                            tree_posterior, tree_terms, weighted_ess)
from dvae.references import RANKINGS, build_pool, tower_bank, tower_loss, tower_posterior_prime

VAL_ANCHORS_PER_POOL = 64  # sampled mode: anchors scored per val pool (all D in enumerated mode)
MAX_AUTO_SAMPLED_BATCH = 64  # auto-sized B never goes above this in sampled mode
NO_GRAD_CHUNK_POSITIONS = 250_000  # reference scoring without gradient runs in chunks of this many positions


def per_position_bytes(args):
    # Calibrated on MPS: ~5.5 GB for 256 images x 1023 tree nodes at d_model=64 with 2 token layers.
    return 21e3 * (args.d_model / 64) * (args.n_tok_layers / 2)


def plan(args):
    """Decide enumerated vs sampled, the anchor count and the memory estimate; refuse what won't fit."""
    V, K, D, S = args.vocab_size, args.k_max, args.num_refs, args.num_samples
    args.enumerate = S >= V**K
    with_grad = args.estimator != "reinforce"
    pos = per_position_bytes(args)
    if args.enumerate:
        nodes = K if V == 1 else (V**K - 1) // (V - 1)
        args.batch_size = args.batch_size or (D if args.ref_ranking == "uniform" else D // 4)
        grad_rows = D if with_grad else args.batch_size
        gb = (grad_rows * nodes * pos + (D - grad_rows) * nodes * pos * 0.1 + 40 * D * V**K) / 1e9
        codes = f"all {V**K} codes enumerated ({nodes} prefix nodes per image)"
    else:
        # Without gradient the reference scoring is chunked, so per-anchor memory is only the kept log-probs.
        per_anchor = S * D * K * ((pos + 12 * V) if with_grad else 64)
        cap = min(MAX_AUTO_SAMPLED_BATCH, D if args.ref_ranking == "uniform" else D // 4)
        chunk_peak = 0 if with_grad else NO_GRAD_CHUNK_POSITIONS * pos * 0.2  # transient, one chunk at a time
        budget = 0.8 * args.max_mem_gb * 1e9 - chunk_peak
        args.batch_size = args.batch_size or max(1, min(cap, int(budget / per_anchor)))
        gb = (args.batch_size * per_anchor + chunk_peak) / 1e9
        codes = f"S={S} sampled codes per anchor at temperature {args.temperature:g}"
    if args.batch_size > D:
        raise SystemExit("--batch-size can't exceed --num-refs (the anchors are part of the pool)")
    print(f"encoder objective: {args.estimator} gradient, {codes}, B={args.batch_size} anchors, D={D} refs "
          f"({args.ref_ranking}), ~{gb:.1f} GB per step")
    if gb > args.max_mem_gb:
        raise SystemExit(f"~{gb:.0f} GB per step (> --max-mem-gb {args.max_mem_gb:g}); lower --batch-size, "
                         f"--num-refs or --num-samples")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    add_train_args(p, out="checkpoints/encoder.pt", batch_size=None, lr=3e-4, epochs=30, patience=3)
    p.add_argument("--estimator", choices=ESTIMATORS, default="full")
    p.add_argument("--num-samples", type=int, default=16, help="S codes per anchor; >= V**k_max enumerates all codes")
    p.add_argument("--temperature", type=float, default=1.0, help="q' = q at this temperature (codes are drawn from q')")
    p.add_argument("--baseline", choices=["loo", "none"], default="loo",
                   help="leave-one-out baseline over the S samples for the REINFORCE term")
    p.add_argument("--num-refs", type=int, default=256, help="D: pool size, anchors included")
    p.add_argument("--ref-ranking", choices=RANKINGS, default="uniform")
    p.add_argument("--ref-uniform-frac", type=float, default=0.25, help="share of extra refs drawn uniformly")
    p.add_argument("--towers-dim", type=int, default=64, help="embedding size for --ref-ranking two-towers")
    p.add_argument("--towers-refresh", type=int, default=200, help="steps between recomputing the image bank")
    p.add_argument("--vocab-size", type=int, default=2)
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--k-dist", choices=["uniform", "geometric", "custom"], default="uniform")
    p.add_argument("--k-geom-ratio", type=float, default=0.7)
    p.add_argument("--k-weights", default=None, help="comma-separated p(k) for --k-dist custom")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-img-layers", type=int, default=2)
    p.add_argument("--n-tok-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--max-mem-gb", type=float, default=8.0,
                   help="refuse to start if the estimated activation memory per step exceeds this")
    args = parse_train_args(p)
    if args.estimator not in ESTIMATORS or not hasattr(args, "ref_ranking"):
        raise SystemExit("this checkpoint was saved by an older train_encoder.py and can't be resumed")
    plan(args)

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    print(f"device {device}")
    train, val = load_mnist("train", args.data_root, device), load_mnist("val", args.data_root, device)
    towers_dim = args.towers_dim if args.ref_ranking == "two-towers" else None
    model = Encoder(args.vocab_size, args.k_max, args.d_model, args.n_img_layers, args.n_tok_layers, args.n_heads,
                    towers_dim).to(device)
    print(f"encoder params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    weights = k_weights(args.k_max, args.k_dist, args.k_geom_ratio, args.k_weights).to(device)
    gen = torch.Generator().manual_seed(args.seed)
    V, K, B, D, S = args.vocab_size, args.k_max, args.batch_size, args.num_refs, args.num_samples

    def prefix_indices(codes):
        """codes [B, K] -> [B, K]: index of each prefix z_1:k among the V**k nodes at depth k."""
        idx, out = torch.zeros_like(codes[:, 0]), []
        for t in range(K):
            idx = idx * V + codes[:, t]
            out.append(idx)
        return torch.stack(out, 1)
    towers_state = {"bank": None, "steps": 0}

    def sample_codes(x_anchors, temperature, generator=None):
        with torch.no_grad():
            memory = model.memory(x_anchors).repeat_interleave(S, 0)
            return model.tokens.sample(len(memory), memory, generator=generator, temperature=temperature)

    def loss_on_pool(x_pool, self_pos, ref_log_w, estimator, temperature, codes=None):
        """Surrogate, per-k values, per-k effective refs, and (codes, log q) for the towers' target."""
        dist = anchor_distances(x_pool.flatten(1), self_pos)
        if args.enumerate:
            if estimator == "reinforce":
                with torch.no_grad():
                    tree = model.prefix_logprobs(x_pool)
                anchor_tree = model.prefix_logprobs(x_pool[self_pos])
            else:
                tree = model.prefix_logprobs(x_pool)
                anchor_tree = [lq[self_pos] for lq in tree]
            posteriors = tree_posterior(tree, ref_log_w, dist, self_pos)
            terms = tree_terms(anchor_tree, posteriors)
            ess = [e for _, e in posteriors]
            tower_codes = codes  # greedy codes, if the caller made them
            tower_target = None
            if codes is not None:
                nodes = prefix_indices(codes)
                tower_target = torch.stack([tree[k + 1][:, nodes[:, k]].T for k in range(K)], -1).detach()
        else:
            nA = len(self_pos)
            memory = model.memory(x_pool)
            anchor_memory = memory[self_pos].repeat_interleave(S, 0)
            assert codes is not None
            logits = model.tokens.logits(codes[:, :-1], model.tokens.memory_kvs(anchor_memory)).float()
            pick = codes.unsqueeze(-1)
            logq = logits.log_softmax(-1).gather(-1, pick).squeeze(-1).view(nA, S, K)
            logq_prime = (logits.detach() / temperature).log_softmax(-1).gather(-1, pick).squeeze(-1).view(nA, S, K)
            if estimator == "reinforce" or not torch.is_grad_enabled():
                with torch.no_grad():
                    step = max(1, NO_GRAD_CHUNK_POSITIONS // (D * K))
                    logq_refs = torch.cat([model.tokens.log_prob_pairwise(codes[c:c + step], memory)
                                           for c in range(0, len(codes), step)])
            else:
                logq_refs = model.tokens.log_prob_pairwise(codes, memory)
            logq_refs = logq_refs.view(nA, S, D, K)
            F, ess_all = sampled_posterior(logq_refs, ref_log_w, dist, self_pos)
            terms = sampled_terms(logq, logq_prime, F, args.baseline == "loo")
            ess = [ess_all[..., k] for k in range(K)]
            tower_codes = codes.view(nA, S, K)[:, 0]
            tower_target = logq_refs[:, 0].cumsum(-1).detach()  # [nA, D, K]: log q(z_1:k | y_j)
        surrogate, per_k = combine(terms, weights, estimator)
        return surrogate, per_k, weighted_ess(terms, ess), tower_codes, tower_target

    def epoch_batches(epoch):
        for anchor_idx in batches(len(train), B, True, device, gen):
            # the seed is drawn here so skipping steps on resume replays the same pools
            yield anchor_idx, int(torch.randint(2**31 - 1, (1,), generator=gen))

    def step_loss(batch):
        anchor_idx, seed = batch
        pool_gen = torch.Generator().manual_seed(seed)
        xa = to_float(train[anchor_idx])
        codes = None if args.enumerate else sample_codes(xa, args.temperature)
        posterior_prime = None
        if args.ref_ranking == "two-towers":
            assert model.towers is not None
            if towers_state["bank"] is None or towers_state["steps"] % args.towers_refresh == 0:
                towers_state["bank"] = tower_bank(model.towers, train)
            towers_state["steps"] += 1
            if args.enumerate:
                codes = proposal_codes = model.sample(xa, mode="greedy")
            else:
                assert codes is not None
                proposal_codes = codes.view(B, S, K)[:, 0]
            posterior_prime = tower_posterior_prime(model.towers, towers_state["bank"], proposal_codes, weights)
        pool_idx, ref_log_w = build_pool(anchor_idx, len(train), D, pool_gen, posterior_prime, args.ref_uniform_frac)
        x_pool = to_float(train[pool_idx])
        self_pos = torch.arange(B, device=device)
        loss, per_k, ess, tower_codes, tower_target = loss_on_pool(
            x_pool, self_pos, ref_log_w, args.estimator, args.temperature, codes)
        logs = {"per_k": per_k.tolist(), "ess": ess.tolist()}
        if model.towers is not None and tower_target is not None:
            t_loss = tower_loss(model.towers, x_pool, tower_codes, tower_target, ref_log_w, self_pos, weights)
            loss = loss + t_loss - t_loss.detach()  # trains the towers without changing the reported value
            logs["tower_loss"] = t_loss.item()
        return loss, logs

    def evaluate():
        # Fixed pools of D uniform val images and fixed code draws from q, independent of the training settings.
        val_gen = torch.Generator(device=device).manual_seed(1234)
        rows = []
        for idx in batches(len(val), D, False, device):
            x_pool = to_float(val[idx])
            ref_log_w = torch.zeros(D, device=device)
            n_anchors = D if args.enumerate else min(D, VAL_ANCHORS_PER_POOL)
            for chunk in batches(n_anchors, D if args.enumerate else B, False, device, drop_last=False):
                codes = None if args.enumerate else sample_codes(x_pool[chunk], 1.0, val_gen)
                _, per_k, ess, _, _ = loss_on_pool(x_pool, chunk, ref_log_w, "full", 1.0, codes)
                k0 = anchor_distances(x_pool.flatten(1), chunk).sum(1).mean() / (D - 1)
                rows.append((len(chunk), per_k, ess, k0))
        n = sum(r[0] for r in rows)
        per_k = torch.stack([r[0] * r[1] for r in rows]).sum(0) / n
        ess = torch.stack([r[0] * r[2] for r in rows]).sum(0) / n
        k0 = torch.stack([r[0] * r[3] for r in rows]).sum() / n
        return {"loss": (weights * per_k).sum().item(), "k0": k0.item(), "per_k": per_k.tolist(), "ess": ess.tolist()}

    fit(model, "encoder", args, epoch_batches, step_loss, evaluate, gen)


if __name__ == "__main__":
    main()
