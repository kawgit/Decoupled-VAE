"""Unconditional pixel-space epsilon DDPM baseline, independent of the tree models.

Residual U-Net, sinusoidal time embeddings, attention at selected resolutions, fixed posterior variance.
The sampler supports full ancestral DDPM and deterministic DDIM previews. Images live in [-1, 1].
References: Ho et al., https://github.com/hojonathanho/diffusion ; Song et al., https://github.com/ermongroup/ddim .
This is a local implementation, not a reproduction of a published checkpoint or FID result.
"""

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def norm(channels):
    return nn.GroupNorm(math.gcd(channels, 32), channels)


def zero_module(module):
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


def time_embedding(t: Tensor, width: int):
    half = width // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    phase = t.float()[:, None] * frequencies[None]
    return F.pad(torch.cat([phase.sin(), phase.cos()], -1), (0, width % 2))


class ResBlock(nn.Module):
    def __init__(self, incoming, outgoing, time_width, dropout):
        super().__init__()
        self.first = nn.Sequential(norm(incoming), nn.SiLU(), nn.Conv2d(incoming, outgoing, 3, padding=1))
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_width, outgoing))
        self.last = nn.Sequential(norm(outgoing), nn.SiLU(), nn.Dropout(dropout),
                                  nn.Conv2d(outgoing, outgoing, 3, padding=1))
        self.skip = nn.Identity() if incoming == outgoing else nn.Conv2d(incoming, outgoing, 1)
        zero_module(self.last[-1])

    def forward(self, x, embedding):
        h = self.first(x) + self.time(embedding)[:, :, None, None]
        return self.skip(x) + self.last(h)


class Attention(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = norm(channels)
        self.qkv = nn.Conv1d(channels, 3 * channels, 1)
        self.out = nn.Conv1d(channels, channels, 1)
        zero_module(self.out)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x).flatten(2)).reshape(B, 3, self.heads, C // self.heads, H * W).unbind(1)
        h = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.out(h.transpose(-1, -2).reshape(B, C, H * W)).view(B, C, H, W)


class TimeBlock(nn.Module):
    def __init__(self, incoming, outgoing, time_width, dropout, attention):
        super().__init__()
        self.res = ResBlock(incoming, outgoing, time_width, dropout)
        self.attention = Attention(outgoing) if attention else nn.Identity()

    def forward(self, x, embedding):
        return self.attention(self.res(x, embedding))


class UNet(nn.Module):
    def __init__(self, width=128, multipliers=(1, 2, 2, 2), blocks=2, dropout=0.1, attention=(16,)):
        super().__init__()
        multipliers, attention = tuple(multipliers), tuple(attention)
        self.config = dict(width=width, multipliers=multipliers, blocks=blocks, dropout=dropout, attention=attention)
        self.width = width
        tw = 4 * width
        self.time = nn.Sequential(nn.Linear(width, tw), nn.SiLU(), nn.Linear(tw, tw))
        self.input = nn.Conv2d(3, width, 3, padding=1)
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        skips, ch, resolution = [width], width, 32
        for level, multiple in enumerate(multipliers):
            for _ in range(blocks):
                self.down.append(TimeBlock(ch, width * multiple, tw, dropout, resolution in attention))
                ch = width * multiple
                skips.append(ch)
            if level != len(multipliers) - 1:
                self.down.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                skips.append(ch)
                resolution //= 2
        self.middle = TimeBlock(ch, ch, tw, dropout, True)
        self.middle_last = ResBlock(ch, ch, tw, dropout)
        for level in reversed(range(len(multipliers))):
            for _ in range(blocks + 1):
                self.up.append(TimeBlock(ch + skips.pop(), width * multipliers[level], tw, dropout,
                                         resolution in attention))
                ch = width * multipliers[level]
            if level:
                self.up.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                             nn.Conv2d(ch, ch, 3, padding=1)))
                resolution *= 2
        assert not skips
        self.output = nn.Sequential(norm(ch), nn.SiLU(), nn.Conv2d(ch, 3, 3, padding=1))
        zero_module(self.output[-1])

    def forward(self, x, t):
        embedding = self.time(time_embedding(t, self.width))
        h = self.input(x)
        skips = [h]
        for layer in self.down:
            h = layer(h, embedding) if isinstance(layer, TimeBlock) else layer(h)
            skips.append(h)
        h = self.middle_last(self.middle(h, embedding), embedding)
        for layer in self.up:
            h = layer(torch.cat([h, skips.pop()], 1), embedding) if isinstance(layer, TimeBlock) else layer(h)
        return self.output(h)


class Diffusion:
    def __init__(self, steps=1000, schedule="linear", device: str | torch.device = "cpu"):
        self.steps, self.schedule = steps, schedule
        if schedule == "linear":
            beta = torch.linspace(1e-4, 0.02, steps, dtype=torch.float64)
        elif schedule == "cosine":
            t = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
            abar = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
            beta = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
        else:
            raise ValueError(schedule)
        alpha = 1 - beta
        abar = alpha.cumprod(0)
        previous = F.pad(abar[:-1], (1, 0), value=1)
        self.abar = abar.float().to(device)
        self.mean_x0 = (beta * previous.sqrt() / (1 - abar)).float().to(device)
        self.mean_xt = ((1 - previous) * alpha.sqrt() / (1 - abar)).float().to(device)
        self.variance = (beta * (1 - previous) / (1 - abar)).float().to(device)

    def noisy(self, x, t, noise):
        a = self.abar[t, None, None, None]
        return a.sqrt() * x + (1 - a).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, n, generator, sampler="ddpm", sampling_steps=None, amp=None):
        from contextlib import nullcontext

        device = self.abar.device
        x = torch.randn(n, 3, 32, 32, device=device, generator=generator)
        if sampler == "ddpm":
            if sampling_steps not in (None, self.steps):
                raise ValueError("Ancestral DDPM uses every diffusion step; use DDIM for fewer steps")
            times = list(reversed(range(self.steps)))
        elif sampler == "ddim":
            count = sampling_steps or 100
            if not 1 <= count <= self.steps:
                raise ValueError("DDIM sampling steps must be in 1..diffusion steps")
            times = torch.linspace(self.steps - 1, 0, count).long().tolist()
        else:
            raise ValueError(sampler)
        model.eval()
        for i, t in enumerate(times):
            with amp() if amp else nullcontext():
                epsilon = model(x, torch.full((n,), t, device=device, dtype=torch.long)).float()
            a = self.abar[t]
            x0 = ((x - (1 - a).sqrt() * epsilon) / a.sqrt()).clamp(-1, 1)
            if sampler == "ddpm":
                x = self.mean_x0[t] * x0 + self.mean_xt[t] * x
                if t:
                    x = x + self.variance[t].sqrt() * torch.randn(x.shape, device=device, generator=generator)
            else:
                previous = self.abar[times[i + 1]] if i + 1 < len(times) else a.new_ones(())
                # Recompute epsilon consistently with the clipped clean-image prediction.
                epsilon = (x - a.sqrt() * x0) / (1 - a).sqrt()
                x = previous.sqrt() * x0 + (1 - previous).sqrt() * epsilon
        return ((x.clamp(-1, 1) + 1) / 2).float()
