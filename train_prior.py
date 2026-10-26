"""Stage 3: train the prior p(z) on codes from the frozen encoder (max likelihood of z ~ q(z|x), x ~ data)."""

import argparse
import math

import torch

from dvae.common import add_common_args, add_mode_args, add_train_args, fit, get_device, load_checkpoint, parse_train_args
from dvae.data import batches, load_mnist, to_float
from dvae.models import Prior


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    add_train_args(p, out="checkpoints/prior.pt", batch_size=256, lr=1e-3, epochs=30, patience=3)
    p.add_argument("--encoder", default="checkpoints/encoder.pt")
    add_mode_args(p, "encoder")
    p.add_argument("--k-max", type=int, default=32, help="code length: how many of the encoder's tokens the prior models")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    args = parse_train_args(p)

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    encoder, _ = load_checkpoint(args.encoder, "encoder", device)
    model = Prior(encoder.V, args.k_max, encoder.n_slices, args.d_model, args.n_layers, args.n_heads).to(device)
    print(f"device {device}, prior params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    train, val = load_mnist("train", args.data_root, device), load_mnist("val", args.data_root, device)
    mode = dict(mode=args.encoder_mode, top_p=args.encoder_top_p)

    val_gen = torch.Generator(device=device).manual_seed(1234)
    with torch.no_grad():
        val_codes = torch.cat([encoder.sample(to_float(val[idx]), args.k_max, generator=val_gen, **mode)
                               for idx in batches(len(val), 500, False, device, drop_last=False)])
        val_slices = encoder.slice_path(val_codes)

    gen = torch.Generator().manual_seed(args.seed)

    def epoch_batches(epoch):
        return batches(len(train), args.batch_size, True, device, gen)

    def step_loss(idx):
        with torch.no_grad():
            codes = encoder.sample(to_float(train[idx]), args.k_max, **mode)
            slices = encoder.slice_path(codes)
        return -model.log_prob(codes, slices).sum(1).mean()

    def evaluate():
        nll = -model.log_prob(val_codes, val_slices).sum(1).mean().item()
        return {"loss": nll, "nats_per_code": nll, "bits_per_code": nll / math.log(2)}

    fit(model, "prior", args, epoch_batches, step_loss, evaluate, gen, encoder_path=args.encoder)


if __name__ == "__main__":
    main()
