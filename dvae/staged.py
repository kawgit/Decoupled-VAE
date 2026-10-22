"""The staged encoder: one network per code position.

Stage t (t = 0 .. k_max-1) predicts token z_{t+1} from the image and the prefix z_1..z_t. Each stage has
several output heads, one per candidate partition it was trained on, and a selector network that picks a
head from the prefix alone. Built by train_encoder.py.

Codes are indexed as base-V numbers with z_1 most significant, so node j at depth t has children
j*V .. j*V+V-1 at depth t+1.
"""

from typing import cast

import torch
import torch.nn as nn

from .models import apply_mode


def node_ids(prefix, vocab_size):
    """prefix [N, t] -> [N] index of the prefix among the V**t nodes at depth t (0 when t = 0)."""
    ids = torch.zeros(prefix.shape[0], dtype=torch.long, device=prefix.device)
    for s in range(prefix.shape[1]):
        ids = ids * vocab_size + prefix[:, s]
    return ids


def all_prefixes(depth, vocab_size, device):
    """Every length-depth prefix, in node order -> [V**depth, depth]."""
    idx = torch.arange(vocab_size**depth, device=device)
    powers = vocab_size ** torch.arange(depth - 1, -1, -1, device=device)
    return (idx[:, None] // powers) % vocab_size


class StageNet(nn.Module):
    """Image + prefix -> logits for every head [N, n_heads, V]. The prefix modulates image features (FiLM)."""

    def __init__(self, vocab_size, depth, n_heads, width=128):
        super().__init__()
        self.V, self.depth, self.n_heads = vocab_size, depth, n_heads
        self.trunk = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.GELU(),  # 14x14
            nn.Conv2d(64, 64, 4, stride=2, padding=1), nn.GELU(),  # 7x7
            nn.Flatten(), nn.Linear(64 * 49, width), nn.GELU(),
        )
        slots = max(depth, 1)  # depth 0 has no prefix; keep the modules so every stage has the same layout
        self.prefix_emb = nn.Embedding(vocab_size, width)
        self.prefix_pos = nn.Parameter(torch.randn(slots, width) * 0.02)
        self.prefix_mlp = nn.Sequential(nn.Linear(slots * width, width), nn.GELU(), nn.Linear(width, 2 * width))
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, n_heads * vocab_size))

    def features(self, x):
        """x: float [N, 1, 28, 28] in [0, 1] -> [N, width]; independent of the prefix, so it can be shared."""
        return self.trunk(x - 0.5)

    def logits(self, features, prefix):
        """features [N, width], prefix [N, depth] -> [N, n_heads, V]."""
        h = features
        if self.depth > 0:
            e = (self.prefix_emb(prefix) + self.prefix_pos).flatten(1)
            gamma, beta = self.prefix_mlp(e).chunk(2, -1)
            h = h * (1 + gamma) + beta
        return self.head(h).view(-1, self.n_heads, self.V)

    def forward(self, x, prefix):
        return self.logits(self.features(x), prefix)


class SelectorNet(nn.Module):
    """Predicts, from the prefix z alone, an image's marginal decompression loss under each of n_heads partitions.

    The encoder uses the partition with the lowest prediction, so every image with the same prefix uses the
    same partition and the new token means the same thing for all of them.
    """

    def __init__(self, vocab_size, depth, n_heads, width=128):
        super().__init__()
        self.depth = depth
        slots = max(depth, 1)  # depth 0: a constant input, so the prediction is one learned vector
        self.emb = nn.Embedding(vocab_size, width)
        self.pos = nn.Parameter(torch.randn(slots, width) * 0.02)
        self.mlp = nn.Sequential(nn.Linear(slots * width, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.out = nn.Linear(width, n_heads)

    def forward(self, prefix):
        """prefix [N, depth] -> predicted loss per partition [N, n_heads]."""
        if self.depth == 0:
            e = self.pos.flatten().expand(prefix.shape[0], -1)
        else:
            e = (self.emb(prefix) + self.pos).flatten(1)
        return self.out(self.mlp(e))

    def choose(self, prefix):
        return self(prefix).argmin(-1)


class StagedEncoder(nn.Module):
    """q(z|x) as a chain of StageNets, each with a SelectorNet choosing its head from the prefix.

    stages: one dict per position with n_heads, width, selector_width. Same interface the other scripts use.
    """

    def __init__(self, vocab_size=2, k_max=0, stages=()):
        super().__init__()
        stages = list(stages)
        assert k_max == len(stages), "k_max is the number of built stages"
        self.config = dict(vocab_size=vocab_size, k_max=k_max, stages=stages)
        self.V = vocab_size
        self.nets = nn.ModuleList(StageNet(vocab_size, t, s["n_heads"], s["width"]) for t, s in enumerate(stages))
        self.selectors = nn.ModuleList(SelectorNet(vocab_size, t, s["n_heads"], s["selector_width"])
                                       for t, s in enumerate(stages))

    @property
    def k_max(self):
        return len(self.nets)

    @property
    def stage_nets(self) -> list[StageNet]:
        return cast(list[StageNet], list(self.nets))

    @property
    def selector_nets(self) -> list[SelectorNet]:
        return cast(list[SelectorNet], list(self.selectors))

    def stage_logits(self, t, features, prefix):
        """Selected-head logits for position t: [N, V]."""
        logits = self.stage_nets[t].logits(features, prefix)
        head = self.selector_nets[t].choose(prefix)
        return logits[torch.arange(len(logits), device=logits.device), head]

    def log_prob(self, x, codes, mode="actual", top_p=0.9):
        """Per-token log-probs of codes [N, k_max] -> [N, k_max]."""
        out = []
        for t, net in enumerate(self.stage_nets):
            logp = apply_mode(self.stage_logits(t, net.features(x), codes[:, :t]), mode, top_p)
            out.append(logp.gather(-1, codes[:, t:t + 1]).squeeze(-1))
        return torch.stack(out, 1)

    @torch.no_grad()
    def sample(self, x, mode="actual", top_p=0.9, generator=None):
        codes = torch.zeros(x.shape[0], 0, dtype=torch.long, device=x.device)
        for t, net in enumerate(self.stage_nets):
            logp = apply_mode(self.stage_logits(t, net.features(x), codes), mode, top_p)
            nxt = torch.multinomial(logp.exp(), 1, generator=generator)
            codes = torch.cat([codes, nxt], 1)
        return codes

    def prefix_logprobs(self, x, mode="actual", top_p=0.9, chunk=65536, depth=None):
        """Enumerate the code tree: entry t is [N, V**t], the log-prob of every length-t prefix, for t up to
        depth (default k_max)."""
        N = x.shape[0]
        out = [torch.zeros(N, 1, device=x.device)]
        for t, net in enumerate(self.stage_nets[:depth]):
            width = self.V**t
            feats = net.features(x)  # computed once per image, reused for every prefix
            prefixes = all_prefixes(t, self.V, x.device)
            logp = torch.empty(N, width, self.V, device=x.device)
            for start in range(0, N * width, chunk):
                pairs = torch.arange(start, min(start + chunk, N * width), device=x.device)
                img, node = pairs // width, pairs % width
                logp.view(-1, self.V)[pairs] = apply_mode(self.stage_logits(t, feats[img], prefixes[node]), mode, top_p)
            out.append((out[-1].unsqueeze(-1) + logp).flatten(1))
        return out
