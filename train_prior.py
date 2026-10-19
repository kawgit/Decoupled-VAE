"""Stage 3: train the prior p(z) on codes from the frozen encoder (max likelihood of z ~ q(z|x), x ~ data)."""

import argparse
import math

import torch

from dvae.common import add_common_args, add_mode_args, add_train_args, fit, get_device, load_checkpoint, save_checkpoint
from dvae.data import batches, load_mnist, to_float
from dvae.models import Prior


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    add_train_args(p, out="checkpoints/prior.pt", batch_size=256, lr=1e-3, epochs=30, patience=3)
    p.add_argument("--encoder", default="checkpoints/encoder.pt")
    add_mode_args(p, "encoder")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--max-steps-per-epoch", type=int, default=None, help="for smoke tests")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    encoder, _ = load_checkpoint(args.encoder, "encoder", device)
    cfg = encoder.config
    model = Prior(cfg["vocab_size"], cfg["k_max"], args.d_model, args.n_layers, args.n_heads).to(device)
    print(f"device {device}, prior params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    train, val = load_mnist("train", args.data_root, device), load_mnist("val", args.data_root, device)
    mode = dict(mode=args.encoder_mode, top_p=args.encoder_top_p)

    val_gen = torch.Generator(device=device).manual_seed(1234)
    val_codes = torch.cat([encoder.sample(to_float(val[idx]), generator=val_gen, **mode)
                           for idx in batches(len(val), 500, False, device, drop_last=False)])

    gen = torch.Generator().manual_seed(args.seed)

    def train_epoch(epoch):
        for i, idx in enumerate(batches(len(train), args.batch_size, True, device, gen)):
            if args.max_steps_per_epoch and i >= args.max_steps_per_epoch:
                break
            codes = encoder.sample(to_float(train[idx]), **mode)
            yield -model.log_prob(codes).sum(1).mean()

    def evaluate():
        nll = -model.log_prob(val_codes).sum(1).mean().item()
        return {"loss": nll, "nats_per_code": nll, "bits_per_code": nll / math.log(2)}

    def save(metrics, history):
        save_checkpoint(args.out, "prior", model, encoder_path=args.encoder, args=vars(args), metrics=metrics,
                        history=history)

    fit(model, args, train_epoch, evaluate, save)
    print(f"saved best prior to {args.out}")


if __name__ == "__main__":
    main()
