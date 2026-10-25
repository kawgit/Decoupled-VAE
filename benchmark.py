"""Benchmark test-set likelihood of prior + decoder(s).

The prior and decoder read the slice the encoder's selector asks at each position, so the encoder (--encoder)
is always loaded. When the code space is small enough to enumerate, we report the exact negative
log-likelihood -log sum_z p(z) p(x|z), and the negative ELBO under q(z|x) (any q gives a valid bound, so
--encoder-mode greedy/topp are fine too), split into distortion and rate.

When it is too large, every number is estimated with the encoder:
the top --beam codes per image under q(z|x), found by beam search, are summed exactly, and --samples codes are
drawn from q restricted to the codes outside them. Then

  p(x) ~= sum_{z in top} p(z) p(x|z) + (1/S) sum_s p(z_s) p(x|z_s) / r(z_s),

where r is the probability each sample was drawn with (q / q(outside the top)). This estimate of p(x) is
unbiased, so -log of it (nll_is) is an upper bound on the NLL on average, tightening as --beam and --samples
grow. The top codes alone give a guaranteed upper bound, nll_top_bound = -log sum_{z in top} p(z) p(x|z). The
-ELBO, distortion and rate are expectations under q estimated the same way (exact over the top codes, sampled
outside them). The decoder and prior run once per distinct code across all test images.

Each decoder checkpoint is scored on the data variant it was trained for: "gray" on 8-bit MNIST,
"binary" on the test set binarized with a fixed seed. Numbers are given in nats/image and bits/dim.
"""

import argparse
import json
import math
import os
import time
from typing import NamedTuple

import torch

from dvae.common import add_common_args, add_mode_args, check_encoder, check_same_codes, get_device, load_checkpoint
from dvae.data import NUM_PIXELS, batches, binary_target, gray_target, load_mnist, to_float
from dvae.models import apply_mode

MAX_CODES = 1 << 16
DEFAULT_ENCODER = "checkpoints/encoder.pt"


def all_codes(V, k_max, device):
    """[V**k_max, k_max] in the tree's node order (z_1 most significant)."""
    idx = torch.arange(V**k_max, device=device)
    powers = V ** torch.arange(k_max - 1, -1, -1, device=device)
    return (idx[:, None] // powers) % V


@torch.no_grad()
def decoder_log_likelihoods(decoder, codes, slices, target, code_chunk, image_chunk):
    """log p(x|z) for every test image and code (with its slices) -> [N, C]."""
    out = torch.empty(len(target), len(codes), device=target.device)
    pix = torch.arange(NUM_PIXELS, device=target.device)[None]
    for c0 in range(0, len(codes), code_chunk):
        logits = decoder(codes[c0:c0 + code_chunk], slices[c0:c0 + code_chunk])
        if decoder.likelihood == "gray":
            logp = logits.log_softmax(-1)  # [c, 784, 256]
            for i0 in range(0, len(target), image_chunk):
                t = target[i0:i0 + image_chunk]
                out[i0:i0 + image_chunk, c0:c0 + code_chunk] = logp[:, pix, t].sum(-1).T
        else:
            lp1, lp0 = torch.nn.functional.logsigmoid(logits), torch.nn.functional.logsigmoid(-logits)
            out[:, c0:c0 + code_chunk] = target @ lp1.T + (1 - target) @ lp0.T
    return out


class Estimate(NamedTuple):
    """What the estimators need per test image when the code space is too large to enumerate."""
    B: int  # top codes per image
    S: int  # sampled codes per image
    codes: torch.Tensor  # distinct codes [U, k]
    pair_code: torch.Tensor  # each (image, code) pair's index among them [N * (B + S)]
    pair_img: torch.Tensor  # each pair's image [N * (B + S)]
    log_q: torch.Tensor  # log q(z|x) of the pairs [N, B + S]
    log_pz: torch.Tensor  # log p(z) of the pairs [N, B + S]
    sample_log_r: torch.Tensor  # log-probability each sample was drawn with [N, S]
    q_rest: torch.Tensor  # q-mass outside the top codes [N]


class Progress:
    """A progress line at most every 10 seconds."""

    def __init__(self, what):
        self.what, self.t0, self.last = what, time.time(), time.time()

    def __call__(self, done, total):
        if time.time() - self.last > 10:
            self.last = time.time()
            print(f"  {self.what}: {done}/{total}, {self.last - self.t0:.0f}s", flush=True)


def next_log_q(encoder, t, feats, prefix, args):
    """log q(z_t+1 = v | x, prefix) for every token v -> [n, V]."""
    return apply_mode(encoder.stage_logits(t, feats, prefix), args.encoder_mode, args.encoder_top_p)


@torch.no_grad()
def top_codes_and_samples(encoder, test, k, B, S, args, device):
    """For every test image: its top-B codes under q(z|x) (beam search) and S codes sampled from q restricted
    to the codes outside them.

    Returns top codes [N, B', k], their log q [N, B'] (B' = min(B, V**k)), samples [N, S, k], their log q
    [N, S] and the log-probability under which each sample was actually drawn [N, S]. The sampler walks the
    code tree choosing each child in proportion to the q-mass of its subtree outside the top codes, so the
    top codes are never drawn; using the actual sampling probability keeps the estimate unbiased even where
    rounding makes it differ slightly from q / q(rest).
    """
    V = encoder.V
    gen = torch.Generator(device=device).manual_seed(args.seed)
    rows = max(B, S, 1)
    out = {key: [] for key in ("top", "top_log_q", "samples", "sample_log_q", "sample_log_r")}
    progress = Progress("beam search and sampling (images)")
    for idx in batches(len(test), max(1, args.encoder_chunk // rows), False, device, drop_last=False):
        x, n = to_float(test[idx]), len(idx)
        feats = encoder.features(x)  # every slice's logits, shared by all prefixes
        # beam search
        beam, beam_lq = torch.zeros(n, 1, 0, dtype=torch.long, device=device), torch.zeros(n, 1, device=device)
        for t in range(k):
            b = beam.shape[1]
            img = torch.arange(n, device=device).repeat_interleave(b)
            lq = beam_lq[:, :, None] + next_log_q(encoder, t, feats[img], beam.flatten(0, 1), args).view(n, b, V)
            keep = min(B, b * V)
            beam_lq, pick = lq.flatten(1).topk(keep, 1)
            parent, token = pick // V, pick % V
            beam = torch.cat([beam.gather(1, parent[:, :, None].expand(-1, -1, t)), token[:, :, None]], 2)
        out["top"].append(beam)
        out["top_log_q"].append(beam_lq)
        # sampling outside the top codes
        if S > 0:
            img = torch.arange(n, device=device).repeat_interleave(S)
            top, top_lq = beam[img], beam_lq[img]  # [nS, B', k], [nS, B']
            prefix = torch.zeros(n * S, 0, dtype=torch.long, device=device)
            lq, lr = torch.zeros(n * S, device=device), torch.zeros(n * S, device=device)
            for t in range(k):
                logp = next_log_q(encoder, t, feats[img], prefix, args)  # [nS, V]
                on_path = (top[:, :, :t] == prefix[:, None, :]).all(-1)  # top codes inside this node's subtree
                child = top[:, :, t]
                count = torch.zeros(n * S, V, device=device).scatter_add_(1, child, on_path.float())
                # q-mass of each child's subtree taken by top codes, relative to the child's own q-mass
                rel = torch.where(on_path, (top_lq - (lq[:, None] + logp.gather(1, child))).clamp(max=0).exp(), 0.0)
                taken = torch.zeros(n * S, V, device=device).scatter_add_(1, child, rel.nan_to_num(0.0))
                full = count >= V ** (k - t - 1)  # every code in the child's subtree is a top code
                weight = logp.exp() * (1 - taken).clamp(min=0) * ~full
                empty = weight.sum(1, keepdim=True) <= 0
                weight = torch.where(empty, (~full).float(), weight)
                probs = weight / weight.sum(1, keepdim=True)
                v = torch.multinomial(probs, 1, generator=gen).squeeze(1)
                lr += probs.gather(1, v[:, None]).squeeze(1).log()
                lq += logp.gather(1, v[:, None]).squeeze(1)
                prefix = torch.cat([prefix, v[:, None]], 1)
            out["samples"].append(prefix.view(n, S, k))
            out["sample_log_q"].append(lq.view(n, S))
            out["sample_log_r"].append(lr.view(n, S))
        progress(int(idx[-1]) + 1, len(test))
    N = len(test)
    if S == 0:
        empty_codes = torch.zeros(N, 0, k, dtype=torch.long, device=device)
        return (torch.cat(out["top"]), torch.cat(out["top_log_q"]), empty_codes,
                torch.zeros(N, 0, device=device), torch.zeros(N, 0, device=device))
    return tuple(torch.cat(out[key]) for key in ("top", "top_log_q", "samples", "sample_log_q", "sample_log_r"))


@torch.no_grad()
def distinct_codes(codes):
    """codes [M, k] -> (distinct codes [U, k], index of each row's code among them [M])."""
    uniq, inv = torch.unique(codes.cpu(), dim=0, return_inverse=True)
    return uniq.to(codes.device), inv.to(codes.device)


@torch.no_grad()
def prior_log_probs(prior, codes, slices, chunk):
    """log p(z) of codes [U, k] with their slices [U, k] -> [U]."""
    return torch.cat([prior.log_prob(codes[i:i + chunk], slices[i:i + chunk]).sum(1)
                      for i in range(0, len(codes), chunk)])


@torch.no_grad()
def pair_log_likelihoods(decoder, codes, slices, pair_code, pair_img, target, code_chunk, pair_chunk):
    """log p(x|z) for (image, code) pairs, decoding each distinct code [U, k] (with its slices) once -> [M]."""
    out = torch.empty(len(pair_code), device=codes.device)
    order = pair_code.argsort()
    sorted_code = pair_code[order]
    pix = torch.arange(NUM_PIXELS, device=codes.device)[None]
    progress = Progress(f"decoding {len(codes)} distinct codes for {len(pair_code)} (image, code) pairs")
    for c0 in range(0, len(codes), code_chunk):
        logits = decoder(codes[c0:c0 + code_chunk], slices[c0:c0 + code_chunk])
        lo, hi = torch.searchsorted(sorted_code, torch.tensor([c0, c0 + code_chunk], device=codes.device)).tolist()
        if decoder.likelihood == "gray":
            logp = logits.log_softmax(-1)  # [c, 784, 256]
        else:
            lp1, lp0 = torch.nn.functional.logsigmoid(logits), torch.nn.functional.logsigmoid(-logits)
        for p0 in range(lo, hi, pair_chunk):
            sel = order[p0:min(p0 + pair_chunk, hi)]
            local, tgt = pair_code[sel] - c0, target[pair_img[sel]]
            if decoder.likelihood == "gray":
                out[sel] = logp[local[:, None], pix, tgt].sum(-1)
            else:
                out[sel] = (tgt * lp1[local] + (1 - tgt) * lp0[local]).sum(-1)
        progress(min(c0 + code_chunk, len(codes)), len(codes))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--prior", default="checkpoints/prior.pt")
    p.add_argument("--decoder", nargs="+", default=["checkpoints/decoder_gray.pt"])
    p.add_argument("--encoder", default=DEFAULT_ENCODER,
                   help="supplies the slices the prior and decoder read; also gives the ELBO under its q(z|x)")
    add_mode_args(p, "encoder")
    p.add_argument("--binarize-seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="score only the first N test images")
    p.add_argument("--code-chunk", type=int, default=64)
    p.add_argument("--image-chunk", type=int, default=500)
    p.add_argument("--beam", type=int, default=16,
                   help="top codes per image summed exactly when the code space is too large to enumerate")
    p.add_argument("--samples", type=int, default=64,
                   help="codes per image sampled outside the top codes when the code space is too large to enumerate")
    p.add_argument("--encoder-chunk", type=int, default=4096, help="encoder rows per pass when estimating")
    p.add_argument("--pair-chunk", type=int, default=4096, help="(image, code) pairs per scoring pass when estimating")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    device = get_device(args.device)
    prior, _ = load_checkpoint(args.prior, "prior", device)
    V, k_max = prior.config["vocab_size"], prior.config["k_max"]
    exact = V**k_max <= MAX_CODES
    if not exact:
        if args.beam >= V**k_max:
            args.samples = 0  # the beam holds every code: nothing is left to sample
        print(f"{V}**{k_max} codes is too many to enumerate exactly: estimating from the top {args.beam} codes "
              f"per image under {args.encoder} plus {args.samples} sampled outside them")
    test = load_mnist("test", args.data_root, device)
    if args.limit:
        test = test[: args.limit]
    N = len(test)

    est: Estimate | None = None
    with torch.no_grad():
        encoder, _ = load_checkpoint(args.encoder, "encoder", device)
        check_encoder(encoder, prior.config, "prior")
        if exact:
            codes = all_codes(V, k_max, device)
            code_slices = encoder.slice_path(codes)
            log_pz = prior.prefix_logprobs(encoder)[k_max]  # [C]
            log_q = torch.cat([encoder.prefix_logprobs(to_float(test[idx]), k_max, args.encoder_mode,
                                                       args.encoder_top_p)[k_max]
                               for idx in batches(N, 128, False, device, drop_last=False)])
        else:
            top, top_lq, smp, smp_lq, smp_lr = top_codes_and_samples(encoder, test, k_max, args.beam, args.samples,
                                                                     args, device)
            B, S = top.shape[1], smp.shape[1]
            uniq, pair_code = distinct_codes(torch.cat([top, smp], 1).flatten(0, 1))
            print(f"{len(uniq)} distinct codes among {len(pair_code)} (image, code) pairs", flush=True)
            est = Estimate(B, S, uniq, pair_code, torch.arange(N, device=device).repeat_interleave(B + S),
                           torch.cat([top_lq, smp_lq], 1),
                           prior_log_probs(prior, uniq, encoder.slice_path(uniq), args.pair_chunk)[pair_code]
                           .view(N, B + S), smp_lr,
                           (1 - top_lq.exp().sum(1)).clamp(min=0))

    rows = []
    for path in args.decoder:
        decoder, _ = load_checkpoint(path, "decoder", device)
        check_same_codes(decoder.config, prior.config, f"{path} vs prior")
        if decoder.likelihood == "gray":
            variant, target = "mnist-8bit", gray_target(test)
        else:
            gen = torch.Generator(device=device).manual_seed(args.binarize_seed)
            variant, target = f"mnist-binarized(seed={args.binarize_seed})", binary_target(test, gen)
        row = {"decoder": path, "variant": variant, "num_images": N}
        elbo_terms = None  # (distortion, rate), each [N]
        if exact:
            log_px_z = decoder_log_likelihoods(decoder, codes, code_slices, target, args.code_chunk,
                                               args.image_chunk)  # [N, C]
            nll = -torch.logsumexp(log_px_z + log_pz, 1)  # exact -log p(x), [N]
            row["nll_nats"] = nll.mean().item()
            q = log_q.exp()
            safe_log_q = torch.where(q > 0, log_q, torch.zeros_like(log_q))
            distortion = -(q * torch.where(q > 0, log_px_z, torch.zeros_like(log_px_z))).sum(1)
            rate = (q * (safe_log_q - torch.where(q > 0, log_pz, torch.zeros_like(log_q)))).sum(1)  # KL(q || p)
            elbo_terms = distortion, rate
        else:
            assert est is not None
            B, S = est.B, est.S
            log_px_z = pair_log_likelihoods(decoder, est.codes, encoder.slice_path(est.codes), est.pair_code,
                                            est.pair_img, target, args.code_chunk, args.pair_chunk).view(N, B + S)
            log_joint = log_px_z + est.log_pz  # log p(z) p(x|z)
            top_mass = torch.logsumexp(log_joint[:, :B], 1)  # log sum over the top codes: <= log p(x)
            log_px = top_mass
            if S > 0:  # plus the importance-sampled rest, under the probability each sample was actually drawn
                rest = torch.logsumexp(log_joint[:, B:] - est.sample_log_r, 1) - math.log(S)
                log_px = torch.logaddexp(top_mass, rest)
            # q-weight of each pair in expectations under q: exact for the top codes, q / r / S for the samples
            weight = torch.cat([est.log_q[:, :B].exp(), (est.log_q[:, B:] - est.sample_log_r).exp() / max(S, 1)], 1)
            safe = weight > 0

            def expect(f):
                return (weight * torch.where(safe, f, 0.0)).sum(1)

            row.update(nll_is_nats=(-log_px).mean().item(), nll_top_bound_nats=(-top_mass).mean().item(),
                       beam=B, samples=S, q_outside_beam=est.q_rest.mean().item())
            if S > 0 or est.q_rest.max() <= 1e-6:  # the ELBO needs all of q: without samples, only if the beam holds it
                elbo_terms = expect(-log_px_z), expect(est.log_q - est.log_pz)
        if elbo_terms is not None:
            distortion, rate = elbo_terms
            row.update(neg_elbo_nats=(distortion + rate).mean().item(), distortion_nats=distortion.mean().item(),
                       rate_nats=rate.mean().item(), encoder=args.encoder, encoder_mode=args.encoder_mode)
        for key in [k for k in row if k.endswith("_nats")]:
            row[key.replace("_nats", "_bpd")] = row[key] / (NUM_PIXELS * math.log(2))
        rows.append(row)
        print(json.dumps(row, indent=2))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "benchmark.json"), "w") as f:
        json.dump({"prior": args.prior, "codes": f"{V}^{k_max}", "exact": exact, "results": rows}, f, indent=2)
    nll_key = "nll" if exact else "nll_is"
    lines = ["| variant | NLL (nats/img) | NLL (bpd) | -ELBO (nats/img) | -ELBO (bpd) | rate (nats) | baseline |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        elbo = (f"{r['neg_elbo_nats']:.2f} | {r['neg_elbo_bpd']:.4f} | {r['rate_nats']:.2f}"
                if "neg_elbo_nats" in r else "– | – | –")
        lines.append(f"| {r['variant']} | {r[nll_key + '_nats']:.2f} | {r[nll_key + '_bpd']:.4f} | {elbo} | |")
    if not exact:
        lines.append(f"\nEstimated with {args.encoder} ({args.encoder_mode}): the top {args.beam} codes per image "
                     f"under q are summed exactly and {args.samples} codes are sampled from q outside them. The NLL "
                     f"is an unbiased estimate of p(x) under the log (an upper bound on average); the top codes alone "
                     f"give a guaranteed upper bound: "
                     + ", ".join(f"{r['variant']} {r['nll_top_bound_nats']:.2f} nats" for r in rows) + ".")
    with open(os.path.join(args.out_dir, "benchmark.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out_dir}/benchmark.json and benchmark.md")


if __name__ == "__main__":
    main()
