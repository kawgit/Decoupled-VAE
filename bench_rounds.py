"""Time train_rounds.py's steps under its speedups, one added at a time (cudnn autotuning, bfloat16 autocast,
channels_last convolutions), and check that bfloat16 training follows float32's losses. On a CUDA device; the
models are freshly initialized at the default sizes, the codes come from round 0's table encoder.

Run from the repo: python bench_rounds.py
"""

import argparse
import copy
import time

import torch

from dvae.artifacts import artifact
from dvae.data import get_device, load_mnist
from dvae.deltatree import pixels
from dvae.logs import RunLog
from dvae.rounds import CodeEncoder, CodePrior, Decoder, DecoderMean, TableEncoder, kmeans_sse, route, sample_codes
from train_rounds import decoder_sse, prior_bits


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depths", type=int, default=24)
    p.add_argument("--images", type=int, default=16384)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=40, help="timed steps per setting")
    p.add_argument("--parity-steps", type=int, default=200)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    device, K, V = get_device(args.device), args.depths, 2
    log = RunLog(artifact("run.log"), device)
    log(f"=== bench_rounds.py {vars(args)} on {torch.cuda.get_device_name(device) if device.type == 'cuda' else device}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    X = pixels(load_mnist("train")[:args.images]).to(device)
    mean = X.mean(0)
    table = TableEncoder(X[torch.randperm(len(X), device=device)[:K * V]].view(K, V, -1)).to(device)
    tree, _ = route(table, X, mean, K, V)

    def models(channels_last):
        torch.manual_seed(1)
        dims = dict(K=K, V=V, channels_last=channels_last)
        enc = CodeEncoder(**dims).to(device)
        enc.mean.copy_(mean)
        return enc, CodePrior(**dims).to(device), DecoderMean(**dims).to(device)

    enc0, prior0, dec0 = models(False)
    codes = sample_codes(table, prior0.eval(), Decoder(dec0).eval(), mean, args.images, K, V, bf16=False)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def time_steps(model, loss, bf16):
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        model.train()
        gen = torch.Generator().manual_seed(0)
        batches = [torch.randint(0, args.images, (args.batch,), generator=gen).to(device)
                   for _ in range(args.steps + 5)]
        for i, idx in enumerate(batches):
            if i == 5:
                sync()
                began = time.time()
            with torch.autocast(device.type, torch.bfloat16, enabled=bf16):
                out = loss(model, idx).float()
            opt.zero_grad(set_to_none=True)
            out.backward()
            opt.step()
        sync()
        return (time.time() - began) / args.steps * 1000

    losses = {
        "encoder": lambda m, idx: kmeans_sse(m, codes.leaf[idx]) / len(idx),
        "prior": lambda m, idx: prior_bits(m, tree, idx).mean(),
        "decoder": lambda m, idx: decoder_sse(m, tree, X, idx).mean(),
    }
    settings = [("tf32", False, False, False), ("+ cudnn autotune", True, False, False),
                ("+ bf16 autocast", True, True, False), ("+ channels_last", True, True, True)]
    for name, bench, bf16, cl in settings:
        torch.backends.cudnn.benchmark = bench
        enc, prior, dec = models(cl)
        times = {stage: time_steps(model, losses[stage], bf16)
                 for stage, model in (("encoder", enc), ("prior", prior), ("decoder", dec))}
        sync()
        began = time.time()
        sample_codes(CodeEncoder(K, V, channels_last=cl).to(device).eval(), prior.eval(), Decoder(dec).eval(), mean,
                     args.images, K, V, bf16=bf16)
        sync()
        log(f"{name:>17}: ms per step: encoder {times['encoder']:.1f}, prior {times['prior']:.1f}, decoder "
            f"{times['decoder']:.1f}; drawing {args.images} codes {time.time() - began:.2f}s")

    torch.backends.cudnn.benchmark = True
    for name, bf16 in (("float32", False), ("bf16", True)):
        enc = copy.deepcopy(enc0)
        opt = torch.optim.AdamW(enc.parameters(), lr=3e-4)
        gen = torch.Generator().manual_seed(0)
        trace = []
        for step in range(args.parity_steps):
            idx = torch.randint(0, args.images, (args.batch,), generator=gen).to(device)
            with torch.autocast(device.type, torch.bfloat16, enabled=bf16):
                out = losses["encoder"](enc, idx).float()
            opt.zero_grad(set_to_none=True)
            out.backward()
            opt.step()
            if step % 50 == 0 or step == args.parity_steps - 1:
                trace.append(f"{step}: {out.item():.4f}")
        log(f"encoder loss, {name}: " + ", ".join(trace))


if __name__ == "__main__":
    main()
