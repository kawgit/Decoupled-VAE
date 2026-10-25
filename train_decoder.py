"""Stage 2: train the decoder p(x|z) on codes from the frozen encoder.

--likelihood gray models 8-bit pixels (bits/dim); --likelihood binary models dynamically binarized
pixels (nats). The encoder always sees the grayscale image.
"""

import argparse
import math

import torch

from dvae.common import add_common_args, add_mode_args, add_train_args, fit, get_device, load_checkpoint, parse_train_args
from dvae.data import NUM_PIXELS, batches, binary_target, gray_target, load_mnist, to_float
from dvae.models import Decoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    add_train_args(p, out=None, batch_size=128, lr=1e-3, epochs=30, patience=3)
    p.add_argument("--encoder", default="checkpoints/encoder.pt")
    p.add_argument("--likelihood", choices=["gray", "binary"], default="gray")
    p.add_argument("--k-max", type=int, default=8, help="code length: how many of the encoder's tokens the decoder reads")
    add_mode_args(p, "encoder")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--channels", type=int, default=128)
    args = parse_train_args(p)
    args.out = args.out or f"checkpoints/decoder_{args.likelihood}.pt"

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    encoder, _ = load_checkpoint(args.encoder, "encoder", device)
    model = Decoder(encoder.V, args.k_max, encoder.n_slices, args.likelihood, args.d_model, args.n_layers,
                    args.n_heads, args.channels).to(device)
    print(f"device {device}, decoder params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    train, val = load_mnist("train", args.data_root, device), load_mnist("val", args.data_root, device)
    mode = dict(mode=args.encoder_mode, top_p=args.encoder_top_p)

    def target(images, generator=None):
        return gray_target(images) if args.likelihood == "gray" else binary_target(images, generator)

    # Fixed val codes and binarization so val loss is comparable across epochs.
    val_gen = torch.Generator(device=device).manual_seed(1234)
    val_codes, val_targets = [], []
    with torch.no_grad():
        for idx in batches(len(val), 500, False, device, drop_last=False):
            codes = encoder.sample(to_float(val[idx]), args.k_max, generator=val_gen, **mode)
            val_codes.append((codes, encoder.slice_path(codes)))
            val_targets.append(target(val[idx], val_gen))

    gen = torch.Generator().manual_seed(args.seed)

    def epoch_batches(epoch):
        return batches(len(train), args.batch_size, True, device, gen)

    def step_loss(idx):
        with torch.no_grad():
            codes = encoder.sample(to_float(train[idx]), args.k_max, **mode)
            slices = encoder.slice_path(codes)
        return -model.log_prob(model(codes, slices), target(train[idx])).mean() / NUM_PIXELS

    def evaluate():
        nll = sum(-model.log_prob(model(*c), t).sum().item() for c, t in zip(val_codes, val_targets)) / len(val)
        return {"loss": nll / NUM_PIXELS, "nats_per_image": nll, "bits_per_dim": nll / (NUM_PIXELS * math.log(2))}

    fit(model, "decoder", args, epoch_batches, step_loss, evaluate, gen, encoder_path=args.encoder)


if __name__ == "__main__":
    main()
