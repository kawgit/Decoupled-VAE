"""Generation QA for the categorical decoder (train_categorical_decoder.py): codes sampled down the autoencoder's
tree (each side from --prior, else the autoencoder's own shares), then each code's pixels drawn from the decoder's
p(x | code).

In --grid x --grid squares, every one from the same codes: samples_means.png, each code's predicted mean image
(what the autoencoder alone decodes it to); samples_pixels.png, one image drawn from p(x | code), every pixel's
grey level drawn independently; samples_mode.png, every pixel's most likely level; real.png, real test images.

Run from the repo: python qa_categorical_decoder.py --decoder artifacts/<job>/decoder.pt
                   [--prior artifacts/<prior job>/prior.pt] [--ckpt artifacts/<run>/model.pt]
"""

import argparse
import os

import torch

from dvae.artifacts import artifact
from dvae.categorical import load_categorical
from dvae.data import get_device, load_mnist
from dvae.deltatree import pixels
from dvae.prior import load_prior
from dvae.sampling import load_autoencoder, sample
from dvae.viz import save_grid


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--decoder", required=True, help="train_categorical_decoder.py's checkpoint")
    p.add_argument("--prior", default=None, help="train_prior.py's checkpoint (default: the autoencoder's shares)")
    p.add_argument("--ckpt", default=None, help="the autoencoder (default: the one the decoder was trained on)")
    p.add_argument("--temperature", type=float, default=1.0, help="the prior's")
    p.add_argument("--grid", type=int, default=8, help="side of the square grids")
    p.add_argument("--out", default=None, help="folder for the pictures (default: the run's artifacts folder)")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="data")
    args = p.parse_args()
    device = get_device(args.device)
    dckpt = torch.load(args.decoder, map_location=device, weights_only=False)
    decoder = load_categorical(dckpt, device)
    ckpt = args.ckpt or dckpt.get("autoencoder")
    assert ckpt, "this decoder's checkpoint doesn't name its autoencoder: give --ckpt"
    decoders, mean, steps, batch, tol = load_autoencoder(ckpt, device)
    K = dckpt["tokens"]
    assert K <= len(decoders), f"the decoder reads {K} tokens, the autoencoder has {len(decoders)}"
    prior = load_prior(torch.load(args.prior, map_location=device, weights_only=False), device) if args.prior else None
    out = args.out or artifact("")
    os.makedirs(out, exist_ok=True)

    n = args.grid ** 2
    gen = torch.Generator().manual_seed(args.seed)
    xs, fr = sample(decoders[:K], mean, n, steps[:K], batch, tol, gen, prior, args.temperature)
    mu = xs[:, -1]
    with torch.no_grad():
        logits = decoder(fr.deltas.to(device).float(), fr.sides.to(device), fr.ms.to(device), mu)  # [n, 256, D]
    probs = logits.softmax(1).permute(0, 2, 1)  # [n, D, 256]
    drawn = torch.multinomial(probs.reshape(-1, 256).cpu(), 1, generator=gen).view(n, -1)
    images = {"samples_means": mu.cpu(), "samples_pixels": (drawn.float() + 0.5) / 256,
              "samples_mode": (logits.argmax(1).cpu().float() + 0.5) / 256}
    source = f"the prior at temperature {args.temperature:g}" if prior is not None else "the autoencoder's own shares"
    for name, x in images.items():
        path = os.path.join(out, f"{name}.png")
        save_grid(x.clamp(0, 1).reshape(-1, 1, 28, 28), path, nrow=args.grid)
        print(f"{path}: {n} codes of {K} tokens sampled from {source}: {name.split('_')[1]}")
    path = os.path.join(out, "real.png")
    save_grid(pixels(load_mnist("test", args.data_root)[:n]).reshape(-1, 1, 28, 28), path, nrow=args.grid)
    print(f"{path}: the first {n} test images, for comparison")


if __name__ == "__main__":
    main()
