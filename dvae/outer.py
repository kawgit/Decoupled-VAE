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
    """The outer encoder (images [B, C_in, H, W] in [0, 1] -> latents [B, channels, H / 4, W / 4]) and decoder."""

    def __init__(self, channels, width=64, image_channels=3):
        super().__init__()
        w = width
        self.config = dict(channels=channels, width=width, image_channels=image_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, w, 3, padding=1), ResBlock(w),
            nn.Conv2d(w, 2 * w, 4, 2, 1), ResBlock(2 * w),  # (H / 2)
            nn.Conv2d(2 * w, 4 * w, 4, 2, 1), ResBlock(4 * w), ResBlock(4 * w),  # (H / 4)
            nn.GroupNorm(8, 4 * w), nn.GELU(), nn.Conv2d(4 * w, channels, 1))
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, 4 * w, 3, padding=1), ResBlock(4 * w), ResBlock(4 * w),
            nn.ConvTranspose2d(4 * w, 2 * w, 4, 2, 1), ResBlock(2 * w),
            nn.ConvTranspose2d(2 * w, w, 4, 2, 1), ResBlock(w),
            nn.GroupNorm(8, w), nn.GELU(), nn.Conv2d(w, image_channels, 3, padding=1))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_outer(folder, device):
    """A train_outer.py run's chosen autoencoder and its latents.pt (latents and the scaling that undoes them)."""
    lat = torch.load(os.path.join(folder, "latents.pt"), map_location="cpu", weights_only=False)
    saved = torch.load(os.path.join(folder, f"outer_c{lat['channels']}.pt"), map_location=device, weights_only=False)
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
