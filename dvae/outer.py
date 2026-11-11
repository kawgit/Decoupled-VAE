"""The outer encoder and decoder (train_outer.py): a convolutional autoencoder from images to a small continuous
latent, and the latents it writes, which the inner encoder (train_inner_components.py --outer) codes."""

import os

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(nn.GroupNorm(8, c), nn.GELU(), nn.Conv2d(c, c, 3, padding=1),
                                  nn.GroupNorm(8, c), nn.GELU(), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.body(x)


class Outer(nn.Module):
    """The outer encoder (images [B, C_in, H, W] in [0, 1] -> latents [B, channels, H / 2 ** downs, W / 2 ** downs])
    and decoder: a stride-2 convolution and a residual block per halving, widths w, 2w, 4w, 4w, ..."""

    def __init__(self, channels, width=64, image_channels=3, downs=2):
        super().__init__()
        widths = [width * min(2 ** i, 4) for i in range(downs + 1)]
        self.config = dict(channels=channels, width=width, image_channels=image_channels, downs=downs)
        enc: list[nn.Module] = [nn.Conv2d(image_channels, widths[0], 3, padding=1), ResBlock(widths[0])]
        for a, b in zip(widths, widths[1:]):
            enc += [nn.Conv2d(a, b, 4, 2, 1), ResBlock(b)]
        enc += [ResBlock(widths[-1]), nn.GroupNorm(8, widths[-1]), nn.GELU(), nn.Conv2d(widths[-1], channels, 1)]
        self.encoder = nn.Sequential(*enc)
        dec: list[nn.Module] = [nn.Conv2d(channels, widths[-1], 3, padding=1), ResBlock(widths[-1]),
                                ResBlock(widths[-1])]
        for a, b in zip(widths[::-1], widths[-2::-1]):
            dec += [nn.ConvTranspose2d(a, b, 4, 2, 1), ResBlock(b)]
        dec += [nn.GroupNorm(8, widths[0]), nn.GELU(), nn.Conv2d(widths[0], image_channels, 3, padding=1)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_outer(path, device):
    """A train_outer.py run's latents (path: the run's folder, for its latents.pt, or one latents file) and the
    autoencoder that made them."""
    file = os.path.join(path, "latents.pt") if os.path.isdir(path) else path
    lat = torch.load(file, map_location="cpu", weights_only=False)
    saved = torch.load(os.path.join(os.path.dirname(file), f"outer_c{lat['channels']}.pt"), map_location=device,
                       weights_only=False)
    model = Outer(**saved["config"]).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, lat


@torch.no_grad()
def decode_latents(model: Outer, lat: dict, Z: torch.Tensor, autocast, batch: int = 1000) -> torch.Tensor:
    """Flattened scaled latents Z [N, C * h * w] (as latents.pt keeps them) -> flattened images [N, D] in [0, 1]."""
    C, h, w = lat["shape"]
    mean, std = lat["mean"].to(Z.device)[None, :, None, None], lat["std"].to(Z.device)[None, :, None, None]
    parts = []
    for i in range(0, len(Z), batch):
        z = Z[i:i + batch].float().view(-1, C, h, w) / lat["scale"] * std + mean
        with autocast():
            parts.append(model.decoder(z).float().clamp(0, 1).flatten(1))
    return torch.cat(parts) if parts else Z.new_zeros(0)
